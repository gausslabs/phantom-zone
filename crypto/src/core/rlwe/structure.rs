use crate::{core::lwe::LweSecretKey, util::distribution::SecretDistribution};
use core::{borrow::Borrow, iter::repeat_with};
use num_traits::{FromPrimitive, Signed};
use phantom_zone_derive::AsSliceWrapper;
use phantom_zone_math::{
    decomposer::DecompositionParam,
    distribution::DistributionSized,
    modulus::ModulusOps,
    poly::automorphism::AutomorphismMap,
    util::{
        as_slice::{AsMutSlice, AsSlice},
        compact::Compact,
        scratch::Scratch,
    },
};
use rand::RngCore;

#[derive(Clone, Copy, Debug, PartialEq, AsSliceWrapper)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RlweSecretKey<S> {
    #[as_slice]
    data: S,
    ring_size: usize,
}

impl<S> RlweSecretKey<S> {
    pub fn new(data: S, ring_size: usize) -> Self {
        Self { data, ring_size }
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }
}

impl<T: Default> RlweSecretKeyOwned<T> {
    pub fn allocate(ring_size: usize) -> Self {
        Self::new(repeat_with(T::default).take(ring_size).collect(), ring_size)
    }

    pub fn sample(ring_size: usize, sk_distribution: SecretDistribution, rng: impl RngCore) -> Self
    where
        T: Signed + FromPrimitive,
    {
        Self::new(sk_distribution.sample_vec(ring_size, rng), ring_size)
    }
}

impl<S> From<RlweSecretKey<S>> for LweSecretKey<S> {
    fn from(sk: RlweSecretKey<S>) -> Self {
        LweSecretKey::new(sk.data)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, AsSliceWrapper)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RlwePlaintext<S> {
    #[as_slice]
    data: S,
    ring_size: usize,
}

impl<S> RlwePlaintext<S> {
    pub fn new(data: S, ring_size: usize) -> Self {
        Self { data, ring_size }
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }
}

impl<T: Default> RlwePlaintextOwned<T> {
    pub fn allocate(ring_size: usize) -> Self {
        Self::new(repeat_with(T::default).take(ring_size).collect(), ring_size)
    }
}

impl<'a, T> RlwePlaintextMutView<'a, T> {
    pub fn scratch(ring_size: usize, scratch: &mut Scratch<'a>) -> Self {
        Self::new(scratch.take_slice(ring_size), ring_size)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, AsSliceWrapper)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RlweCiphertext<S> {
    #[as_slice]
    data: S,
    ring_size: usize,
}

impl<S> RlweCiphertext<S> {
    pub fn new(data: S, ring_size: usize) -> Self {
        Self { data, ring_size }
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }
}

impl<S: AsSlice> RlweCiphertext<S> {
    pub fn ct_size(&self) -> usize {
        self.data.len()
    }

    pub fn a(&self) -> &[S::Elem] {
        self.a_b().0
    }

    pub fn b(&self) -> &[S::Elem] {
        self.a_b().1
    }

    pub fn a_b(&self) -> (&[S::Elem], &[S::Elem]) {
        self.data.split_at_mid()
    }
}

impl<S: AsMutSlice> RlweCiphertext<S> {
    pub fn a_mut(&mut self) -> &mut [S::Elem] {
        self.a_b_mut().0
    }

    pub fn b_mut(&mut self) -> &mut [S::Elem] {
        self.a_b_mut().1
    }

    pub fn a_b_mut(&mut self) -> (&mut [S::Elem], &mut [S::Elem]) {
        self.data.split_at_mid_mut()
    }
}

impl<T: Default> RlweCiphertextOwned<T> {
    pub fn allocate(ring_size: usize) -> Self {
        let ct_size = 2 * ring_size;
        Self::new(repeat_with(T::default).take(ct_size).collect(), ring_size)
    }

    pub fn allocate_eval(ring_size: usize, eval_size: usize) -> Self {
        let ct_size = 2 * eval_size;
        Self::new(repeat_with(T::default).take(ct_size).collect(), ring_size)
    }
}

impl<'a, T> RlweCiphertextMutView<'a, T> {
    pub fn scratch(ring_size: usize, eval_size: usize, scratch: &mut Scratch<'a>) -> Self {
        let ct_size = 2 * eval_size;
        Self::new(scratch.take_slice(ct_size), ring_size)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, AsSliceWrapper)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RlweCiphertextList<S> {
    #[as_slice]
    data: S,
    ring_size: usize,
    ct_size: usize,
}

impl<S> RlweCiphertextList<S> {
    pub fn ct_size(&self) -> usize {
        self.ct_size
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }
}

impl<S: AsSlice> RlweCiphertextList<S> {
    pub fn new(data: S, ring_size: usize, ct_size: usize) -> Self {
        debug_assert_eq!(data.len() % ct_size, 0);
        Self {
            data,
            ring_size,
            ct_size,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.ct_size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = RlweCiphertextView<S::Elem>> {
        let ring_size = self.ring_size();
        let ct_size = self.ct_size();
        self.as_ref()
            .chunks(ct_size)
            .map(move |ct| RlweCiphertext::new(ct, ring_size))
    }

    pub fn chunks(
        &self,
        chunk_size: usize,
    ) -> impl Iterator<Item = RlweCiphertextListView<S::Elem>> {
        let ring_size = self.ring_size();
        let ct_size = self.ct_size();
        self.as_ref()
            .chunks(chunk_size * ct_size)
            .map(move |ct| RlweCiphertextList::new(ct, ring_size, ct_size))
    }
}

impl<S: AsMutSlice> RlweCiphertextList<S> {
    pub fn iter_mut(&mut self) -> impl Iterator<Item = RlweCiphertextMutView<S::Elem>> {
        let ring_size = self.ring_size();
        let ct_size = self.ct_size();
        self.as_mut()
            .chunks_mut(ct_size)
            .map(move |ct| RlweCiphertext::new(ct, ring_size))
    }

    pub fn chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl Iterator<Item = RlweCiphertextListMutView<S::Elem>> {
        let ring_size = self.ring_size();
        let ct_size = self.ct_size();
        self.as_mut()
            .chunks_mut(chunk_size * ct_size)
            .map(move |ct| RlweCiphertextList::new(ct, ring_size, ct_size))
    }
}

impl<T: Default> RlweCiphertextListOwned<T> {
    pub fn allocate(ring_size: usize, n: usize) -> Self {
        let ct_size = 2 * ring_size;
        Self::new(
            repeat_with(T::default).take(n * ct_size).collect(),
            ring_size,
            ct_size,
        )
    }

    pub fn allocate_eval(ring_size: usize, eval_size: usize, n: usize) -> Self {
        let ct_size = 2 * eval_size;
        Self::new(
            repeat_with(T::default).take(n * ct_size).collect(),
            ring_size,
            ct_size,
        )
    }
}

impl<'a, T> RlweCiphertextListMutView<'a, T> {
    pub fn scratch(
        ring_size: usize,
        eval_size: usize,
        n: usize,
        scratch: &mut Scratch<'a>,
    ) -> Self {
        let ct_size = 2 * eval_size;
        Self::new(scratch.take_slice(n * ct_size), ring_size, ct_size)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, AsSliceWrapper)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RlwePublicKey<S>(#[as_slice(nested)] RlweCiphertext<S>);

impl<S: AsSlice> RlwePublicKey<S> {
    pub fn new(ct: RlweCiphertext<S>) -> Self {
        Self(ct)
    }

    pub fn ring_size(&self) -> usize {
        self.0.ring_size()
    }

    pub fn a(&self) -> &[S::Elem] {
        self.0.a()
    }

    pub fn b(&self) -> &[S::Elem] {
        self.0.b()
    }

    pub fn a_b(&self) -> (&[S::Elem], &[S::Elem]) {
        self.0.a_b()
    }

    pub fn ct(&self) -> RlweCiphertextView<S::Elem> {
        self.0.as_view()
    }
}

impl<S: AsMutSlice> RlwePublicKey<S> {
    pub fn a_mut(&mut self) -> &mut [S::Elem] {
        self.0.a_mut()
    }

    pub fn b_mut(&mut self) -> &mut [S::Elem] {
        self.0.b_mut()
    }

    pub fn a_b_mut(&mut self) -> (&mut [S::Elem], &mut [S::Elem]) {
        self.0.a_b_mut()
    }

    pub fn ct_mut(&mut self) -> RlweCiphertextMutView<S::Elem> {
        self.0.as_mut_view()
    }
}

impl<T: Default> RlwePublicKeyOwned<T> {
    pub fn allocate(ring_size: usize) -> Self {
        Self::new(RlweCiphertext::allocate(ring_size))
    }
}

impl<'a, T> RlwePublicKeyMutView<'a, T> {
    pub fn scratch(ring_size: usize, eval_size: usize, scratch: &mut Scratch<'a>) -> Self {
        Self::new(RlweCiphertext::scratch(ring_size, eval_size, scratch))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, AsSliceWrapper)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RlweKeySwitchKey<S> {
    #[as_slice(nested)]
    cts: RlweCiphertextList<S>,
    decomposition_param: DecompositionParam,
}

impl<S> RlweKeySwitchKey<S> {
    pub fn ring_size(&self) -> usize {
        self.cts.ring_size()
    }

    pub fn ct_size(&self) -> usize {
        self.cts.ct_size()
    }

    pub fn decomposition_param(&self) -> DecompositionParam {
        self.decomposition_param
    }
}

impl<S: AsSlice> RlweKeySwitchKey<S> {
    pub fn new(cts: RlweCiphertextList<S>, decomposition_param: DecompositionParam) -> Self {
        debug_assert_eq!(cts.len(), decomposition_param.level);
        Self {
            cts,
            decomposition_param,
        }
    }

    pub fn ct_iter(&self) -> impl Iterator<Item = RlweCiphertextView<S::Elem>> {
        self.cts.iter()
    }
}

impl<S: AsMutSlice> RlweKeySwitchKey<S> {
    pub fn ct_iter_mut(&mut self) -> impl Iterator<Item = RlweCiphertextMutView<S::Elem>> {
        self.cts.iter_mut()
    }
}

impl<T: Default> RlweKeySwitchKeyOwned<T> {
    pub fn allocate(ring_size: usize, decomposition_param: DecompositionParam) -> Self {
        Self::new(
            RlweCiphertextList::allocate(ring_size, decomposition_param.level),
            decomposition_param,
        )
    }

    pub fn allocate_eval(
        ring_size: usize,
        eval_size: usize,
        decomposition_param: DecompositionParam,
    ) -> Self {
        Self::new(
            RlweCiphertextList::allocate_eval(ring_size, eval_size, decomposition_param.level),
            decomposition_param,
        )
    }
}

pub type RlweAutoKeyOwned<T> = RlweAutoKey<Vec<T>, AutomorphismMap>;
pub type RlweAutoKeyView<'a, T> = RlweAutoKey<&'a [T], &'a AutomorphismMap>;
pub type RlweAutoKeyMutView<'a, T> = RlweAutoKey<&'a mut [T], &'a AutomorphismMap>;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "S: serde::Serialize, A: serde::Serialize",
        deserialize = "S: serde::Deserialize<'de>, A: serde::Deserialize<'de>"
    ))
)]
pub struct RlweAutoKey<S, A> {
    ks_key: RlweKeySwitchKey<S>,
    auto_map: A,
}

impl<S, A: Borrow<AutomorphismMap>> RlweAutoKey<S, A> {
    pub fn new(ks_key: RlweKeySwitchKey<S>, auto_map: A) -> Self {
        debug_assert_eq!(ks_key.ring_size(), auto_map.borrow().ring_size());
        Self { ks_key, auto_map }
    }

    pub fn ring_size(&self) -> usize {
        self.ks_key.ring_size()
    }

    pub fn decomposition_param(&self) -> DecompositionParam {
        self.ks_key.decomposition_param()
    }

    pub fn auto_map(&self) -> &AutomorphismMap {
        self.auto_map.borrow()
    }

    pub fn k(&self) -> usize {
        self.auto_map().k()
    }
}

impl<S: AsSlice, A: Borrow<AutomorphismMap>> RlweAutoKey<S, A> {
    pub fn as_view(&self) -> RlweAutoKeyView<S::Elem> {
        RlweAutoKey::new(self.ks_key.as_view(), self.auto_map())
    }

    pub fn ks_key(&self) -> RlweKeySwitchKeyView<S::Elem> {
        self.ks_key.as_view()
    }

    pub fn auto_map_and_ks_key(&self) -> (&AutomorphismMap, RlweKeySwitchKeyView<S::Elem>) {
        (self.auto_map.borrow(), self.ks_key.as_view())
    }

    pub fn cloned(&self) -> RlweAutoKeyOwned<S::Elem>
    where
        S::Elem: Clone,
    {
        RlweAutoKey::new(self.ks_key.cloned(), self.auto_map().clone())
    }

    pub fn compact<M: ModulusOps>(&self, modulus: &M) -> RlweAutoKey<Compact, AutomorphismMap>
    where
        S: AsSlice<Elem = M::Elem>,
    {
        RlweAutoKey::new(self.ks_key.compact(modulus), self.auto_map.borrow().clone())
    }
}

impl<S: AsMutSlice, A: Borrow<AutomorphismMap>> RlweAutoKey<S, A> {
    pub fn as_mut_view(&mut self) -> RlweAutoKeyMutView<S::Elem> {
        RlweAutoKey::new(self.ks_key.as_mut_view(), self.auto_map.borrow())
    }

    pub fn ks_key_mut(&mut self) -> RlweKeySwitchKeyMutView<S::Elem> {
        self.ks_key.as_mut_view()
    }

    pub fn auto_map_and_ks_key_mut(
        &mut self,
    ) -> (&AutomorphismMap, RlweKeySwitchKeyMutView<S::Elem>) {
        (self.auto_map.borrow(), self.ks_key.as_mut_view())
    }
}

impl<T: Default> RlweAutoKeyOwned<T> {
    pub fn allocate(ring_size: usize, decomposition_param: DecompositionParam, k: usize) -> Self {
        Self::new(
            RlweKeySwitchKey::allocate(ring_size, decomposition_param),
            AutomorphismMap::new(ring_size, k),
        )
    }

    pub fn allocate_eval(
        ring_size: usize,
        eval_size: usize,
        decomposition_param: DecompositionParam,
        k: usize,
    ) -> Self {
        Self::new(
            RlweKeySwitchKey::allocate_eval(ring_size, eval_size, decomposition_param),
            AutomorphismMap::new(ring_size, k),
        )
    }
}

impl<A: Borrow<AutomorphismMap>> RlweAutoKey<Compact, A> {
    pub fn uncompact<M: ModulusOps>(&self, modulus: &M) -> RlweAutoKeyOwned<M::Elem> {
        RlweAutoKey::new(
            self.ks_key.uncompact(modulus),
            self.auto_map.borrow().clone(),
        )
    }
}

impl<'from, S: AsSlice, A: Borrow<AutomorphismMap>> From<&'from RlweAutoKey<S, A>>
    for RlweAutoKeyView<'from, S::Elem>
{
    fn from(value: &'from RlweAutoKey<S, A>) -> Self {
        value.as_view()
    }
}

impl<'from, S: AsMutSlice, A: Borrow<AutomorphismMap>> From<&'from mut RlweAutoKey<S, A>>
    for RlweAutoKeyMutView<'from, S::Elem>
{
    fn from(value: &'from mut RlweAutoKey<S, A>) -> Self {
        value.as_mut_view()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, AsSliceWrapper)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeededRlweCiphertext<S> {
    #[as_slice]
    data: S,
    ring_size: usize,
}

impl<S> SeededRlweCiphertext<S> {
    pub fn new(data: S, ring_size: usize) -> Self {
        Self { data, ring_size }
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }
}

impl<S: AsSlice> SeededRlweCiphertext<S> {
    pub fn ct_size(&self) -> usize {
        self.data.len()
    }

    pub fn b(&self) -> &[S::Elem] {
        self.data.as_ref()
    }
}

impl<S: AsMutSlice> SeededRlweCiphertext<S> {
    pub fn b_mut(&mut self) -> &mut [S::Elem] {
        self.data.as_mut()
    }
}

impl<T: Default> SeededRlweCiphertextOwned<T> {
    pub fn allocate(ring_size: usize) -> Self {
        let ct_size = ring_size;
        Self::new(repeat_with(T::default).take(ct_size).collect(), ring_size)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, AsSliceWrapper)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeededRlweCiphertextList<S> {
    #[as_slice]
    data: S,
    ring_size: usize,
    ct_size: usize,
}

impl<S> SeededRlweCiphertextList<S> {
    pub fn ct_size(&self) -> usize {
        self.ct_size
    }

    pub fn ring_size(&self) -> usize {
        self.ring_size
    }
}

impl<S: AsSlice> SeededRlweCiphertextList<S> {
    pub fn new(data: S, ring_size: usize, ct_size: usize) -> Self {
        debug_assert_eq!(data.len() % ct_size, 0);
        Self {
            data,
            ring_size,
            ct_size,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.ct_size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = SeededRlweCiphertextView<S::Elem>> {
        let ring_size = self.ring_size();
        let ct_size = self.ct_size();
        self.as_ref()
            .chunks(ct_size)
            .map(move |ct| SeededRlweCiphertext::new(ct, ring_size))
    }

    pub fn chunks(
        &self,
        chunk_size: usize,
    ) -> impl Iterator<Item = SeededRlweCiphertextListView<S::Elem>> {
        let ring_size = self.ring_size();
        let ct_size = self.ct_size();
        self.as_ref()
            .chunks(chunk_size * ct_size)
            .map(move |ct| SeededRlweCiphertextList::new(ct, ring_size, ct_size))
    }
}

impl<S: AsMutSlice> SeededRlweCiphertextList<S> {
    pub fn iter_mut(&mut self) -> impl Iterator<Item = SeededRlweCiphertextMutView<S::Elem>> {
        let ring_size = self.ring_size();
        let ct_size = self.ct_size();
        self.as_mut()
            .chunks_mut(ct_size)
            .map(move |ct| SeededRlweCiphertext::new(ct, ring_size))
    }

    pub fn chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl Iterator<Item = SeededRlweCiphertextListMutView<S::Elem>> {
        let ring_size = self.ring_size();
        let ct_size = self.ct_size();
        self.as_mut()
            .chunks_mut(chunk_size * ct_size)
            .map(move |ct| SeededRlweCiphertextList::new(ct, ring_size, ct_size))
    }
}

impl<T: Default> SeededRlweCiphertextListOwned<T> {
    pub fn allocate(ring_size: usize, n: usize) -> Self {
        let ct_size = ring_size;
        Self::new(
            repeat_with(T::default).take(n * ct_size).collect(),
            ring_size,
            ct_size,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, AsSliceWrapper)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeededRlwePublicKey<S>(#[as_slice(nested)] SeededRlweCiphertext<S>);

impl<S> SeededRlwePublicKey<S> {
    pub fn new(ct: SeededRlweCiphertext<S>) -> Self {
        Self(ct)
    }

    pub fn ring_size(&self) -> usize {
        self.0.ring_size()
    }
}

impl<S: AsSlice> SeededRlwePublicKey<S> {
    pub fn b(&self) -> &[S::Elem] {
        self.0.b()
    }

    pub fn ct(&self) -> SeededRlweCiphertextView<S::Elem> {
        self.0.as_view()
    }
}

impl<S: AsMutSlice> SeededRlwePublicKey<S> {
    pub fn ct_mut(&mut self) -> SeededRlweCiphertextMutView<S::Elem> {
        self.0.as_mut_view()
    }
}

impl<T: Default> SeededRlwePublicKeyOwned<T> {
    pub fn allocate(ring_size: usize) -> Self {
        Self::new(SeededRlweCiphertext::allocate(ring_size))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, AsSliceWrapper)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeededRlweKeySwitchKey<S> {
    #[as_slice(nested)]
    cts: SeededRlweCiphertextList<S>,
    decomposition_param: DecompositionParam,
}

impl<S> SeededRlweKeySwitchKey<S> {
    pub fn ring_size(&self) -> usize {
        self.cts.ring_size()
    }

    pub fn ct_size(&self) -> usize {
        self.cts.ct_size()
    }

    pub fn decomposition_param(&self) -> DecompositionParam {
        self.decomposition_param
    }
}

impl<S: AsSlice> SeededRlweKeySwitchKey<S> {
    pub fn new(cts: SeededRlweCiphertextList<S>, decomposition_param: DecompositionParam) -> Self {
        debug_assert_eq!(cts.len(), decomposition_param.level);
        Self {
            cts,
            decomposition_param,
        }
    }

    pub fn ct_iter(&self) -> impl Iterator<Item = SeededRlweCiphertextView<S::Elem>> {
        self.cts.iter()
    }
}

impl<S: AsMutSlice> SeededRlweKeySwitchKey<S> {
    pub fn ct_iter_mut(&mut self) -> impl Iterator<Item = SeededRlweCiphertextMutView<S::Elem>> {
        self.cts.iter_mut()
    }
}

impl<T: Default> SeededRlweKeySwitchKeyOwned<T> {
    pub fn allocate(ring_size: usize, decomposition_param: DecompositionParam) -> Self {
        Self::new(
            SeededRlweCiphertextList::allocate(ring_size, decomposition_param.level),
            decomposition_param,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, AsSliceWrapper)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeededRlweAutoKey<S> {
    #[as_slice(nested)]
    ks_key: SeededRlweKeySwitchKey<S>,
    k: usize,
}

impl<S> SeededRlweAutoKey<S> {
    pub fn new(ks_key: SeededRlweKeySwitchKey<S>, k: usize) -> Self {
        debug_assert!(k < 2 * ks_key.ring_size());
        Self { ks_key, k }
    }

    pub fn ring_size(&self) -> usize {
        self.ks_key.ring_size()
    }

    pub fn decomposition_param(&self) -> DecompositionParam {
        self.ks_key.decomposition_param()
    }

    pub fn k(&self) -> usize {
        self.k
    }
}

impl<S: AsSlice> SeededRlweAutoKey<S> {
    pub fn ks_key(&self) -> SeededRlweKeySwitchKeyView<S::Elem> {
        self.ks_key.as_view()
    }
}

impl<S: AsMutSlice> SeededRlweAutoKey<S> {
    pub fn ks_key_mut(&mut self) -> SeededRlweKeySwitchKeyMutView<S::Elem> {
        self.ks_key.as_mut_view()
    }
}

impl<T: Default> SeededRlweAutoKeyOwned<T> {
    pub fn allocate(ring_size: usize, decomposition_param: DecompositionParam, k: usize) -> Self {
        Self::new(
            SeededRlweKeySwitchKey::allocate(ring_size, decomposition_param),
            k,
        )
    }
}
