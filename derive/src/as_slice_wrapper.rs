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
    GenericParam, Index, Result, Token, WhereClause,
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
            GenericParam::Type(ty) => Ok(&ty.ident),
            _ => Err(call_site_err(
                "Lifetime or const generics are not supported yet",
            )),
        })
        .collect::<Result<Vec<_>>>()?;

    let (input_impl_generics, input_type_generics, input_where_clause) =
        input.generics.split_for_impl();
    let input_where_clause = input_where_clause
        .cloned()
        .unwrap_or_else(|| parse(quote!(where)));
    let mut as_view_where_clause = input_where_clause.clone();
    let mut as_mut_view_where_clause = input_where_clause;
    let mut as_view_type_generics = Generics::default();
    let mut as_mut_view_type_generics = Generics::default();
    let mut input_owned_alias_generics = Generics::default();
    let mut input_owned_type_generics = Generics::default();
    let mut input_view_alias_generics = Generics(once(quote!('alias)).collect());
    let mut input_view_type_generics = Generics::default();
    let mut input_mut_view_alias_generics = Generics(once(quote!('alias)).collect());
    let mut input_mut_view_type_generics = Generics::default();
    let mut compact_type_generics = Generics::default();
    let mut compact_where_clause: WhereClause = parse(quote!(where));
    let mut uncompact_type_generics = Generics::default();
    let mut cloned_type_generics = Generics::default();
    let mut cloned_where_clause: WhereClause = parse(quote!(where));
    let mut from_impl_generics = input.generics.clone();
    from_impl_generics.params = once(parse(quote!('from)))
        .chain(from_impl_generics.params)
        .collect();
    let mut from_type_generics = Generics(once(quote!('from)).collect());

    as_slice_generics.iter().for_each(|ident| {
        as_view_where_clause.predicates.push(parse(
            quote! { #ident: phantom_zone_math::util::as_slice::AsSlice },
        ));
        as_mut_view_where_clause.predicates.push(parse(
            quote! { #ident: phantom_zone_math::util::as_slice::AsMutSlice },
        ));
        as_view_type_generics.push(quote! { &[#ident::Elem] });
        as_mut_view_type_generics.push(quote! { &mut [#ident::Elem] });
        compact_type_generics.push(quote! { phantom_zone_math::util::compact::Compact });
        compact_where_clause.predicates.push(parse(
            quote! { #ident: phantom_zone_math::util::as_slice::AsSlice<Elem = M::Elem> },
        ));
        uncompact_type_generics.push(quote! { Vec<M::Elem> });
        cloned_type_generics.push(quote! { Vec<#ident::Elem> });
        cloned_where_clause
            .predicates
            .push(parse(quote! { #ident::Elem: Clone }));

        let elem_ident = {
            let ident = ident.to_string();
            let suffix = ident.trim_start_matches(char::is_alphabetic);
            format_ident!("T{suffix}")
        };
        input_owned_alias_generics.push(quote! { #elem_ident });
        input_owned_type_generics.push(quote! { Vec<#elem_ident> });
        input_view_alias_generics.push(quote! { #elem_ident });
        input_view_type_generics.push(quote! { &'alias [#elem_ident] });
        input_mut_view_alias_generics.push(quote! { #elem_ident });
        input_mut_view_type_generics.push(quote! { &'alias mut [#elem_ident] });
        from_type_generics.push(quote! { #ident::Elem });
    });

    let impl_as_ref_and_as_mut = {
        let nested = *as_slice_fields.first_entry().unwrap().get();
        (as_slice_fields.len() == 1 && !nested)
            .then(|| {
                let field_ident = &field_idents[*as_slice_fields.first_entry().unwrap().key()];
                let generic_ident = as_slice_generics[0];

                quote! {
                    impl #input_impl_generics AsRef<[#generic_ident::Elem]> for #input_ident #input_type_generics #as_view_where_clause {
                        fn as_ref(&self) -> &[#generic_ident::Elem] {
                            self.#field_ident.as_ref()
                        }
                    }

                    impl #input_impl_generics AsMut<[#generic_ident::Elem]> for #input_ident #input_type_generics #as_mut_view_where_clause {
                        fn as_mut(&mut self) -> &mut [#generic_ident::Elem] {
                            self.#field_ident.as_mut()
                        }
                    }
                }
            })
            .unwrap_or_default()
    };

    Ok(quote! {
        pub type #input_owned_ident #input_owned_alias_generics = #input_ident #input_owned_type_generics;
        pub type #input_view_ident #input_view_alias_generics = #input_ident #input_view_type_generics;
        pub type #input_mut_view_ident #input_mut_view_alias_generics = #input_ident #input_mut_view_type_generics;

        impl #input_impl_generics #input_ident #input_type_generics #as_view_where_clause {
            pub fn as_view(&self) -> #input_ident #as_view_type_generics {
                #input_ident {
                    #(#other_field_idents: self.#other_field_idents,)*
                    #(#as_slice_nested_field_idents: self.#as_slice_nested_field_idents.as_view(),)*
                    #(#as_slice_field_idents: self.#as_slice_field_idents.as_ref(),)*
                }
            }

            pub fn cloned(&self) -> #input_ident #cloned_type_generics #cloned_where_clause {
                #input_ident {
                    #(#other_field_idents: self.#other_field_idents,)*
                    #(#as_slice_nested_field_idents: self.#as_slice_nested_field_idents.cloned(),)*
                    #(#as_slice_field_idents: self.#as_slice_field_idents.as_ref().to_vec(),)*
                }
            }

            pub fn compact<M: phantom_zone_math::modulus::ModulusOps>(&self, modulus: &M) -> #input_ident #compact_type_generics #compact_where_clause {
                #input_ident {
                    #(#other_field_idents: self.#other_field_idents,)*
                    #(#as_slice_nested_field_idents: self.#as_slice_nested_field_idents.compact(modulus),)*
                    #(#as_slice_field_idents: phantom_zone_math::util::compact::Compact::from_elems(modulus, &self.#as_slice_field_idents),)*
                }
            }
        }

        impl #input_impl_generics #input_ident #input_type_generics #as_mut_view_where_clause {
            pub fn as_mut_view(&mut self) -> #input_ident #as_mut_view_type_generics {
                #input_ident {
                    #(#other_field_idents: self.#other_field_idents,)*
                    #(#as_slice_nested_field_idents: self.#as_slice_nested_field_idents.as_mut_view(),)*
                    #(#as_slice_field_idents: self.#as_slice_field_idents.as_mut(),)*
                }
            }
        }

        impl #input_ident #compact_type_generics {
            pub fn uncompact<M: phantom_zone_math::modulus::ModulusOps>(&self, modulus: &M) -> #input_ident #uncompact_type_generics {
                #input_ident {
                    #(#other_field_idents: self.#other_field_idents,)*
                    #(#as_slice_nested_field_idents: self.#as_slice_nested_field_idents.uncompact(modulus),)*
                    #(#as_slice_field_idents: self.#as_slice_field_idents.to_elems(modulus),)*
                }
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

        #impl_as_ref_and_as_mut
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
