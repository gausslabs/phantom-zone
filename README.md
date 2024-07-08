**Phantom zone** is similar to the zone where superman gets locked and observes everything outside the zone, but no one outside can see or hear superman. However our phantom zone isn't meant to lock anyone. Instead it's meant to be a new zone in parallel to reality. It's the zone to which you teleport yourself with others, take arbitrary actions, and remember only predefined set of memories when you're back. Think of the zone as a computer that erases itself off of the face of the earth after it returns the output, leaving no trace behind.

**More formally, phantom-zone is a experimental multi-party computation library that uses multi-party fully homomorphic encryption to compute arbitrary functions on private inputs from multiple parties.**

At the moment phantom-zone is pretty limited in its functionality. It offers to write circuits with encrypted 8 bit unsigned integers (referred to as FheUint8) and only supports upto 8 parties. FheUint8 supports the same arithmetic as a regular uint8, with a few exceptions mentioned below. We plan to extend APIs to other signed/unsigned types in the future.

We provide two types of multi-party protocols, both only differ in key-generation procedure. 
1.  **Non-interactive multi-party protocol,** which requires a single shot message from the clients to the server after which the server can evaluate any arbitrary function on encrypted client inputs. 
2. **Interactive multi-party protocol**, a 2 round protocol where in the first round clients interact to generate collective public key and in the second round clients send their server key share to the server, after which server can evaluate any arbitrary function on encrypted client inputs.

Understanding that library is in experimental stage, if you want to use it for your application but find it lacking in features, please don't shy away from opening a issue or getting in touch. We don't want you to hold back those imaginative horses!

## How to use

We provide a brief overview below. Please refer to detailed examples, especially [non_interactive_fheuint8](./examples/non_interactive_fheuint8.rs) and [interactive_fheuint8](./examples/interactive_fheuint8.rs), to understand how to instantiate and run the protocols.

### Non-interactive multi-party

Each client is assigned an `id`, referred to as `user_id`, which denotes serial no. of the client out of total clients participating in the multi-party protocol. After learning their `user_id`, the client uploads their server key share along with encryptions of private inputs in a single shot message to the server. Server can then evaluate any arbitrary function on clients' private inputs. New private inputs can be provided in the future by the fix set of parties that participated in the protocol.

### Interactive multi-party

Like the non-interactive multi-party, each client is assigned `user_id`. After learning their `id`, clients participate in a 2 round protocol. In round 1, clients generate public key shares, share it with each other, and aggregate public key shares to produce the collective public key. In round 2, clients use the collective public key to generate their server key shares and encrypt their private inputs. Server receives server key shares and encryptions of private inputs from each client. Server aggregates the server key shares, after which it can evaluate any arbitrary function on clients' private inputs. New private inputs can be provided in the future by anyone with access to collective public key.

### Multi-party decryption

To decrypt output ciphertext(s) obtained as result of some computation, the clients come online. They download output ciphertext(s) from the server, generate decryption shares, and share it with other parties. Clients, after receiving decryption shares of other parties, aggregate the shares and decrypt the ciphertext(s).

### Parameter selection

We provide parameters to run both multi-party protocols for upto 8 parties.

| $\leq$ # Parties | Interactive multi-party | Non-interactive multi-party |
| ------------ | ----------------------- | --------------------------- |
| 2            | InteractiveLTE2Party    | NonInteractiveLTE2Party     |
| 4            | InteractiveLTE4Party    | NonInteractiveLTE4Party     |
| 8            | InteractiveLTE8Party    | NonInteractiveLTE8Party     |

If you have use-case `> 8` parties, please open an issue. We're willing to find suitable parameters for `> 8` parties.

Parameters supporting `<= N` parties must not be used for multi-party compute between `> N` parties. This will lead to increase in failure probability.

### Feature selection

To use the library for non-interactive multi-party, you must add `non_interactive_mp` feature flag like `--features "non_interactive_mp"`. And to use the library for interactive multi-party you must add `interactive_mp` feature flag like `--features "interactive_mp"`.

### FheUInt8

We provide APIs for all basic arithmetic (+, -, x, /, %) and comparison operations.

All arithmetic operation by default wrap around (i.e. $\mod{256}$ for FheUint8). We also provide `overflow_{add/add_assign}` and `overflow_sub` that returns a flag ciphertext which is set to `True` if addition/subtraction overflowed and to `False` otherwise.

Division operation (/) returns `quotient` and the remainder operation (%) returns `remainder` s.t. `dividend = division x quotient + remainder`. If both `quotient` and `remainder` are required, then `div_rem` can be used. In case of division by zero, [Div by zero error flag](#Div-by-zero-error-flag) will be set and `quotient` will be set to `255` and `remainder` to equal `dividend`.

**Div by zero error flag**

In encrypted domain there's no way to panic upon division by zero at runtime. Instead we set a local flag ciphertext accessible via `div_by_zero_flag()` that stores a boolean ciphertext indicating whether any of the divisions performed during the execution attempted division by zero. Assuming division by zero detection is critical for your application, we recommend decrypting the flag ciphertext along with other output ciphertexts in multi-party decryption procedure.

The div by zero flag is thread local. If you run multiple different FHE circuits in sequence without stopping the thread (i.e. within a single program) you will have to reset div by zero error flag before starting of the next in-sequence circuit execution with `reset_error_flags()`.

Please refer to [div_by_zero](./examples/div_by_zero.rs) example for more details.

**If and else using mux**

Branching in encrypted domain is expensive because the code must execute all the branches. Hence cost grows exponentially with no. of conditional branches. In general we recommend to modify the code to minimise conditional branches. However, if a code cannot be modified to made branchless, we provide `mux` API for FheUint8s. `mux` selects one of the two FheUint8s based on a selector bit. Please refer to [if_and_else](./examples/if_and_else.rs) example for more details.

## Security

> [!WARNING]
> Code has not been audited and we currently do not provide any security guarantees outside of the cryptographic parameters. We don't recommend to deploy it in production or use to handle sensitive data.

All provided parameters are $2^{128}$ ring operations secure according to [lattice estimator](https://github.com/malb/lattice-estimator) and have failure probability of $\leq 2^{-40}$. However, there are two vital points to keep in mind:

1. Users must not generate two different decryption shares for the same ciphertext, as it can lead to key-recovery attacks. To avoid this, we suggest users maintain a local table listing ciphertext against any previously generated decryption share. Then only generate a new decryption share if ciphertext does not exist in the table, otherwise return the existing share. We believe this should be handled by the library and will add support for this in future.
2. Users must not run the MPC protocol more than once for the same application seed and produce different outputs, as it can lead to key-recovery attacks. We believe this should be handled by the library and will add support for this in future.

## Credits

- We thank Barry Whitehat and Brian Lawrence for many helpful discussions.
- We thank Vivek Bhupatiraju and Andrew Lu for for many insightful discussions on fascinating phantom zone applications.
- Non-interactive multi-party RLWE key generation setup is a new protocol designed by Jean Philippe and Janmajaya. We thank Christian Mouchet for the review of the protocol and helpful suggestions.
- We thank Yao Wang for his help with rust compiler troubles. 

## References
1. [Efficient FHEW Bootstrapping with Small Evaluation Keys, and Applications to Threshold Homomorphic Encryption](https://eprint.iacr.org/2022/198.pdf)
2. [Multiparty Homomorphic Encryption from Ring-Learning-with-Errors](https://eprint.iacr.org/2020/304.pdf)