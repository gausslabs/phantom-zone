#[cfg(not(feature = "serde"))]
pub trait Serde {}

#[cfg(not(feature = "serde"))]
impl<T> Serde for T {}

#[cfg(feature = "serde")]
pub trait Serde: serde::Serialize + serde::de::DeserializeOwned {}

#[cfg(feature = "serde")]
impl<T: serde::Serialize + serde::de::DeserializeOwned> Serde for T {}

#[cfg(any(test, feature = "dev"))]
pub mod dev {
    #[cfg(feature = "serde")]
    pub fn assert_serde_eq<T>(value: &T)
    where
        T: core::fmt::Debug + PartialEq + serde::Serialize + serde::de::DeserializeOwned,
    {
        assert_eq!(
            value,
            &bincode::deserialize(&bincode::serialize(value).unwrap()).unwrap()
        )
    }
}
