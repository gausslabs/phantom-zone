extern crate proc_macro;

mod as_slice_wrapper;

#[proc_macro_derive(AsSliceWrapper, attributes(as_slice))]
pub fn as_slice_wrapper(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse(input).unwrap();
    match as_slice_wrapper::derive(&ast) {
        Ok(ok) => ok.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
