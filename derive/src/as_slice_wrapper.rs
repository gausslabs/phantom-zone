use crate::call_site_err;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Index, Result};

pub fn derive(input: &DeriveInput) -> Result<TokenStream> {
    let Data::Struct(data_struct) = &input.data else {
        return Err(call_site_err("Only struct is supported"));
    };

    let fields: Vec<_> = match &data_struct.fields {
        Fields::Unnamed(fields) => fields.unnamed.iter().collect(),
        Fields::Named(fields) => fields.named.iter().collect(),
        _ => return Err(call_site_err("Only non-unit struct is supported")),
    };

    let as_slice_field_idx = fields
        .iter()
        .position(|field| {
            field
                .attrs
                .iter()
                .flat_map(|attr| &attr.path().segments)
                .any(|seg| seg.ident == "as_slice")
        })
        .or_else(|| (fields.len() == 1).then_some(0))
        .ok_or(call_site_err(
            "Struct with multiple fields must have `#[as_slice]` specified",
        ))?;

    let field_idents: Vec<_> = match &data_struct.fields {
        Fields::Unnamed(fields) => (0..fields.unnamed.len())
            .map(|idx| {
                let idx = Index::from(idx);
                quote! { #idx }
            })
            .collect(),
        Fields::Named(fields) => fields
            .named
            .iter()
            .map(|field| {
                let ident = field.ident.as_ref().unwrap();
                quote! { #ident }
            })
            .collect(),
        _ => unreachable!(),
    };
    let as_slice_field_ident = &field_idents[as_slice_field_idx];
    let rest_field_idents: Vec<_> = field_idents
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != as_slice_field_idx)
        .map(|(_, field)| field)
        .collect();

    let input_ident = &input.ident;
    let input_owned_ident = format_ident!("{}Owned", input_ident);
    let input_view_ident = format_ident!("{}View", input_ident);
    let input_mut_view_ident = format_ident!("{}MutView", input_ident);

    Ok(quote! {
        pub type #input_owned_ident<T> = #input_ident<Vec<T>>;
        pub type #input_view_ident<'a, T> = #input_ident<&'a [T]>;
        pub type #input_mut_view_ident<'a, T> = #input_ident<&'a mut [T]>;

        impl<S: AsSlice> #input_ident<S> {
            pub fn as_view(&self) -> #input_view_ident<S::Elem> {
                #input_ident {
                    #(#rest_field_idents: self.#rest_field_idents,)*
                    #as_slice_field_ident: self.#as_slice_field_ident.as_ref(),
                }
            }

            pub fn len(&self) -> usize {
                self.#as_slice_field_ident.len()
            }

            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }
        }

        impl<S: AsMutSlice> #input_ident<S> {
            pub fn as_mut_view(&mut self) -> #input_mut_view_ident<S::Elem> {
                #input_ident {
                    #(#rest_field_idents: self.#rest_field_idents,)*
                    #as_slice_field_ident: self.#as_slice_field_ident.as_mut(),
                }
            }
        }

        impl<S: AsSlice> AsRef<[S::Elem]> for #input_ident<S> {
            fn as_ref(&self) -> &[S::Elem] {
                self.#as_slice_field_ident.as_ref()
            }
        }

        impl<S: AsMutSlice> AsMut<[S::Elem]> for #input_ident<S> {
            fn as_mut(&mut self) -> &mut [S::Elem] {
                self.#as_slice_field_ident.as_mut()
            }
        }
    })
}
