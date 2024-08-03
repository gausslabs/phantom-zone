use core::{
    array::from_fn,
    fmt::Display,
    iter::once,
    ops::{Deref, DerefMut},
};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote, ToTokens};
use std::collections::BTreeMap;
use syn::{
    parse::Parse, parse2, punctuated::Punctuated, Attribute, Data, DeriveInput, Error, Fields,
    GenericArgument, GenericParam, Index, PathArguments, Result, Token, TypeParamBound,
};

pub fn derive(input: &DeriveInput) -> Result<TokenStream> {
    let Data::Struct(data_struct) = &input.data else {
        return Err(call_site_err("Only struct is supported"));
    };

    let fields: Vec<_> = match &data_struct.fields {
        Fields::Unnamed(fields) => fields.unnamed.iter().collect(),
        Fields::Named(fields) => fields.named.iter().collect(),
        _ => return Err(call_site_err("Only non-unit struct is supported")),
    };

    let mut as_slice_fields: BTreeMap<usize, bool> = fields
        .iter()
        .enumerate()
        .flat_map(|(idx, field)| {
            let find = |attr: &Attribute| match attr.path().segments.first() {
                Some(seg) if seg.ident == "as_slice" => match attr.parse_args::<Ident>() {
                    Ok(nested) if nested == "nested" => Some((idx, true)),
                    _ => Some((idx, false)),
                },
                _ => None,
            };
            field.attrs.iter().find_map(find)
        })
        .collect();

    if as_slice_fields.is_empty() {
        if fields.len() == 1 {
            as_slice_fields.insert(0, false);
        } else {
            return Err(call_site_err(
                "Struct with multiple fields must have `#[as_slice]` specified",
            ));
        }
    }

    let field_idents: Vec<_> = match &data_struct.fields {
        Fields::Unnamed(fields) => (0..fields.unnamed.len())
            .map(Index::from)
            .map(|idx| quote!(#idx))
            .collect(),
        Fields::Named(fields) => fields
            .named
            .iter()
            .map(|field| &field.ident)
            .map(|ident| quote!(#ident))
            .collect(),
        _ => unreachable!(),
    };

    let input_ident = &input.ident;
    let [input_owned_ident, input_view_ident, input_mut_view_ident] =
        ["Owned", "View", "MutView"].map(|suffix| format_ident!("{input_ident}{suffix}"));
    let first_as_slice_field_ident = &field_idents[*as_slice_fields.first_entry().unwrap().key()];
    let [as_slice_nested_field_idents, as_slice_field_idents, other_field_idents] =
        field_idents.iter().enumerate().fold(
            from_fn(|_| Vec::new()),
            |mut field_idents, (idx, field_ident)| {
                let kind = match as_slice_fields.get(&idx) {
                    Some(true) => 0,
                    Some(false) => 1,
                    None => 2,
                };
                field_idents[kind].push(field_ident);
                field_idents
            },
        );

    let as_slice_generics = input
        .generics
        .params
        .iter()
        .map(|param| match &param {
            GenericParam::Type(ty) => {
                let ident = &ty.ident;
                let elem_bound = ty
                    .bounds
                    .iter()
                    .flat_map(|bound| match bound {
                        TypeParamBound::Trait(tr) => tr
                            .path
                            .segments
                            .iter()
                            .flat_map(|seg| {
                                if seg.ident == "AsSlice" {
                                    match &seg.arguments {
                                        PathArguments::AngleBracketed(generic) => generic
                                            .args
                                            .iter()
                                            .flat_map(|arg| match arg {
                                                GenericArgument::AssocType(assoc)
                                                    if assoc.ident == "Elem" =>
                                                {
                                                    Some(&assoc.ty)
                                                }
                                                _ => None,
                                            })
                                            .next(),
                                        _ => None,
                                    }
                                } else {
                                    None
                                }
                            })
                            .next(),
                        _ => None,
                    })
                    .next();
                Ok((ident, elem_bound))
            }
            _ => Err(call_site_err(
                "Lifetime or const generics are not supported yet",
            )),
        })
        .collect::<Result<Vec<_>>>()?;

    let first_as_slice_generic_ident = as_slice_generics.first().unwrap().0;

    let (input_impl_generics, input_type_generics, input_where_clause) =
        input.generics.split_for_impl();
    let input_where_clause = input_where_clause
        .cloned()
        .unwrap_or_else(|| parse(quote!(where)));
    let mut as_view_where_clause = input_where_clause.clone();
    let mut as_mut_view_where_clause = input_where_clause;
    let mut as_view_generics = Generics::default();
    let mut as_mut_view_generics = Generics::default();
    let mut input_owned_impl_generics = Generics::default();
    let mut input_owned_type_genercs = Generics::default();
    let mut input_view_impl_generics = Generics(once(quote!('alias)).collect());
    let mut input_view_type_genercs = Generics::default();
    let mut input_mut_view_impl_generics = Generics(once(quote!('alias)).collect());
    let mut input_mut_view_type_genercs = Generics::default();
    let mut from_impl_generics = input.generics.clone();
    from_impl_generics.params = once(parse(quote! { 'from }))
        .chain(from_impl_generics.params)
        .collect();
    let mut from_type_generics = Generics(once(quote!('from)).collect());

    as_slice_generics.iter().for_each(|(ident, elem_bound)| {
        as_view_where_clause
            .predicates
            .push(parse(quote! { #ident: AsSlice }));
        as_mut_view_where_clause
            .predicates
            .push(parse(quote! { #ident: AsMutSlice }));
        as_view_generics.push(quote! { &[#ident::Elem] });
        as_mut_view_generics.push(quote! { &mut [#ident::Elem] });
        if let Some(elem_bound) = elem_bound {
            input_owned_type_genercs.push(quote! { Vec<#elem_bound> });
            input_view_type_genercs.push(quote! { &'alias [#elem_bound] });
            input_mut_view_type_genercs.push(quote! { &'alias mut [#elem_bound] });
        } else {
            let generic_ident = {
                let ident = ident.to_string();
                let suffix = ident.trim_start_matches(char::is_alphabetic);
                format_ident!("T{suffix}")
            };
            input_owned_impl_generics.push(quote! { #generic_ident });
            input_owned_type_genercs.push(quote! { Vec<#generic_ident> });
            input_view_impl_generics.push(quote! { #generic_ident });
            input_view_type_genercs.push(quote! { &'alias [#generic_ident] });
            input_mut_view_impl_generics.push(quote! { #generic_ident });
            input_mut_view_type_genercs.push(quote! { &'alias mut [#generic_ident] });
            from_type_generics.push(quote! { #ident::Elem });
        }
    });

    Ok(quote! {
        pub type #input_owned_ident #input_owned_impl_generics = #input_ident #input_owned_type_genercs;
        pub type #input_view_ident #input_view_impl_generics = #input_ident #input_view_type_genercs;
        pub type #input_mut_view_ident #input_mut_view_impl_generics = #input_ident #input_mut_view_type_genercs;

        impl #input_impl_generics #input_ident #input_type_generics #as_view_where_clause {
            pub fn as_view(&self) -> #input_ident #as_view_generics {
                #input_ident {
                    #(#other_field_idents: self.#other_field_idents,)*
                    #(#as_slice_nested_field_idents: self.#as_slice_nested_field_idents.as_view(),)*
                    #(#as_slice_field_idents: self.#as_slice_field_idents.as_ref(),)*
                }
            }

            pub fn len(&self) -> usize {
                self.#first_as_slice_field_ident.len()
            }

            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }
        }

        impl #input_impl_generics #input_ident #input_type_generics #as_mut_view_where_clause {
            pub fn as_mut_view(&mut self) -> #input_ident #as_mut_view_generics {
                #input_ident {
                    #(#other_field_idents: self.#other_field_idents,)*
                    #(#as_slice_nested_field_idents: self.#as_slice_nested_field_idents.as_mut_view(),)*
                    #(#as_slice_field_idents: self.#as_slice_field_idents.as_mut(),)*
                }
            }
        }

        impl #input_impl_generics AsRef<[#first_as_slice_generic_ident::Elem]> for #input_ident #input_type_generics #as_view_where_clause {
            fn as_ref(&self) -> &[#first_as_slice_generic_ident::Elem] {
                self.#first_as_slice_field_ident.as_ref()
            }
        }

        impl #input_impl_generics AsMut<[#first_as_slice_generic_ident::Elem]> for #input_ident #input_type_generics #as_mut_view_where_clause {
            fn as_mut(&mut self) -> &mut [#first_as_slice_generic_ident::Elem] {
                self.#first_as_slice_field_ident.as_mut()
            }
        }

        impl #from_impl_generics From<&'from #input_ident #input_type_generics> for #input_view_ident #from_type_generics #as_view_where_clause {
            fn from(value: &'from #input_ident #input_type_generics) -> Self {
                value.as_view()
            }
        }

        impl #from_impl_generics From<&'from mut #input_ident #input_type_generics> for #input_mut_view_ident #from_type_generics #as_mut_view_where_clause {
            fn from(value: &'from mut #input_ident #input_type_generics) -> Self {
                value.as_mut_view()
            }
        }
    })
}

fn call_site_err(msg: impl Display) -> Error {
    Error::new(Span::call_site(), msg)
}

fn parse<T: Parse>(ts: TokenStream) -> T {
    parse2(ts).unwrap()
}

#[derive(Default)]
struct Generics(Punctuated<TokenStream, Token![,]>);

impl Deref for Generics {
    type Target = Punctuated<TokenStream, Token![,]>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Generics {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl ToTokens for Generics {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        <Token![<]>::default().to_tokens(tokens);
        self.0.to_tokens(tokens);
        <Token![>]>::default().to_tokens(tokens);
    }
}
