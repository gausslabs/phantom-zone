use crate::call_site_err;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::Parse, Data, DeriveInput, Fields, GenericArgument, GenericParam, Index, PathArguments,
    Result, TypeParamBound, WhereClause,
};

pub fn derive(input: &DeriveInput) -> Result<TokenStream> {
    let fields: Vec<_> = match &input.data {
        Data::Struct(data_struct) if !data_struct.fields.is_empty() => {
            Ok(match &data_struct.fields {
                Fields::Unnamed(fields) => fields.unnamed.iter().collect(),
                Fields::Named(fields) => fields.named.iter().collect(),
                _ => unreachable!(),
            })
        }
        _ => Err(call_site_err("Only struct is supported")),
    }?;

    let as_slice_field_idx = fields
        .iter()
        .position(|field| {
            field.attrs.iter().any(|attr| {
                attr.path()
                    .segments
                    .iter()
                    .any(|seg| seg.ident == "as_slice")
            })
        })
        .or_else(|| (fields.len() == 1).then_some(0))
        .ok_or(call_site_err(
            "Struct with multiple fields must have `#[as_slice]` specified",
        ))?;
    let as_slice_field = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Unnamed(_) => {
                let idx = Index::from(as_slice_field_idx);
                quote! { #idx }
            }
            Fields::Named(fields) => {
                let ident = fields.named[as_slice_field_idx].ident.as_ref().unwrap();
                quote! { #ident }
            }
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };
    let rest_fields: Vec<_> = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Unnamed(fields) => (0..fields.unnamed.len())
                .filter(|idx| *idx != as_slice_field_idx)
                .map(|idx| {
                    let idx = Index::from(idx);
                    quote! { #idx }
                })
                .collect(),
            Fields::Named(fields) => fields
                .named
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != as_slice_field_idx)
                .map(|(_, field)| {
                    let ident = field.ident.as_ref().unwrap();
                    quote! { #ident }
                })
                .collect(),
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };

    let as_slice_elem = input
        .generics
        .params
        .iter()
        .flat_map(|param| match &param {
            GenericParam::Type(ty) => ty
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
                .next(),
            _ => None,
        })
        .next();

    let input_ident = &input.ident;
    let input_owned_ident = format_ident!("{}Owned", input_ident);
    let input_view_ident = format_ident!("{}View", input_ident);
    let input_mut_view_ident = format_ident!("{}MutView", input_ident);

    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let where_clause = where_clause
        .cloned()
        .unwrap_or_else(|| syn::parse2::<WhereClause>(quote! { where }).unwrap());
    let mut as_slice_where_clause = where_clause.clone();
    as_slice_where_clause
        .predicates
        .push_value(syn::parse2(quote! { S: AsSlice<Elem = #as_slice_elem> }).unwrap());
    let mut as_mut_slice_where_clause = where_clause.clone();
    as_mut_slice_where_clause
        .predicates
        .push_value(syn::parse2(quote! { S: AsMutSlice<Elem = #as_slice_elem> }).unwrap());

    if let Some(as_slice_elem) = as_slice_elem {
        Ok(quote! {
            pub type #input_owned_ident = #input_ident<Vec<#as_slice_elem>>;
            pub type #input_view_ident<'a> = #input_ident<&'a [#as_slice_elem]>;
            pub type #input_mut_view_ident<'a> = #input_ident<&'a mut [#as_slice_elem]>;

            impl #impl_generics #input_ident #type_generics #as_slice_where_clause {
                pub fn as_view(&self) -> #input_view_ident {
                    #input_ident {
                        #(#rest_fields: self.#rest_fields,)*
                        #as_slice_field: self.#as_slice_field.as_ref(),
                    }
                }

                pub fn len(&self) -> usize {
                    self.#as_slice_field.len()
                }

                pub fn is_empty(&self) -> bool {
                    self.len() == 0
                }
            }

            impl #impl_generics #input_ident #type_generics #as_mut_slice_where_clause {
                pub fn as_mut_view(&mut self) -> #input_mut_view_ident {
                    #input_ident {
                        #(#rest_fields: self.#rest_fields,)*
                        #as_slice_field: self.#as_slice_field.as_mut(),
                    }
                }
            }

            impl #impl_generics AsRef<[S::Elem]> for #input_ident #type_generics #as_slice_where_clause {
                fn as_ref(&self) -> &[S::Elem] {
                    self.#as_slice_field.as_ref()
                }
            }

            impl #impl_generics AsMut<[S::Elem]> for #input_ident #type_generics #as_mut_slice_where_clause {
                fn as_mut(&mut self) -> &mut [S::Elem] {
                    self.#as_slice_field.as_mut()
                }
            }
        })
    } else {
        Ok(quote! {
            pub type #input_owned_ident<T> = #input_ident<Vec<T>>;
            pub type #input_view_ident<'a, T> = #input_ident<&'a [T]>;
            pub type #input_mut_view_ident<'a, T> = #input_ident<&'a mut [T]>;

            impl<S: AsSlice> #input_ident<S> {
                pub fn as_view(&self) -> #input_view_ident<S::Elem> {
                    #input_ident {
                        #(#rest_fields: self.#rest_fields,)*
                        #as_slice_field: self.#as_slice_field.as_ref(),
                    }
                }

                pub fn len(&self) -> usize {
                    self.#as_slice_field.len()
                }

                pub fn is_empty(&self) -> bool {
                    self.len() == 0
                }
            }

            impl<S: AsMutSlice> #input_ident<S> {
                pub fn as_mut_view(&mut self) -> #input_mut_view_ident<S::Elem> {
                    #input_ident {
                        #(#rest_fields: self.#rest_fields,)*
                        #as_slice_field: self.#as_slice_field.as_mut(),
                    }
                }
            }

            impl<S: AsSlice> AsRef<[S::Elem]> for #input_ident<S> {
                fn as_ref(&self) -> &[S::Elem] {
                    self.#as_slice_field.as_ref()
                }
            }

            impl<S: AsMutSlice> AsMut<[S::Elem]> for #input_ident<S> {
                fn as_mut(&mut self) -> &mut [S::Elem] {
                    self.#as_slice_field.as_mut()
                }
            }
        })
    }
}
