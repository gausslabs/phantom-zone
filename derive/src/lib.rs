extern crate proc_macro;

use core::fmt::Display;
use proc_macro::TokenStream;
use proc_macro2::Span;
use syn::Error;

mod as_slice_wrapper;

fn call_site_err(msg: impl Display) -> Error {
    Error::new(Span::call_site(), msg)
}

#[proc_macro_derive(AsSliceWrapper, attributes(as_slice))]
pub fn as_slice_wrapper(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    match as_slice_wrapper::derive(&ast) {
        Ok(ok) => ok.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
