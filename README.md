Do you recall when superman gets locked in a phantom zone? It's a weird concept where superman can hear and observe things outside the zone but no one can see superman. Well this is what Phantom is about. With Phantom you and your friends can port yourselves to the phantom zone, play games, have conversations, perform any arbitrary compute and only remember the things that you pre-conditioned yourselves to remember when you're back. Think of it as a computer that erases itself off of the face of earth (with cryptographic security) after it returns the output.

More formally, Phantom is a experimental multi-party computation library that uses fully homomorphic encryption to compute arbitrary functions on private inputs from several parties.

At the moment Phantom is pretty limited in its functionality. It offers to write circuits with encrypted 8 bit unsigned integers (referred to as FheUint8). You can work with FheUint8 like any normal Uint8 with a few exceptions mentioned below. We don't plan to just stick with 8 bit types and have plans to extend the APIs to more unsigned / signed types.

We provide two types of multi-party protocols, both only differ in key-generation procedure. First is non-interactive multi-party protocol, which requires a single shot message from the clients to the server after which the server can evaluate any arbitrary function on encrypted client inputs. Second is interactive multi-party protocol. It is a 2 round protocol where in the first round clients interact to generate collective public key and in the second round clients send their server key share to the server, after which server can evaluate any arbitrary function on encrypted client inputs.

Understanding that library is in experimental stage, if you want to use it for your application but find it lacking in features, please don't shy away from opening a issue or getting in touch. We don't want you to hold back those imaginative horses!

## How to use

We only give an overview of the multi-party protocols here. Please refer to detailed examples, especially [non_interactive_fheuint8](./examples/non_interactive_fheuint8.rs) and [interactive_fheuint8](./examples/interactive_fheuint8.rs), to understand how to instantiate and run the protocols.

### Non-interactive multi-party

Each client is assigned an `id`, referred to as `user_id`, which denotes serial no. of the client out of total clients participating in the multi-party protocol. After learning their `user_id`, the client uploads their server key share along with encryptions of private inputs in a single shot message to the server. After receiving messages from each client, server can evaluate any arbitrary function on clients private inputs sent along in the first message, or other inputs provided by clients anytime in future, for the fixed set of clients.

### Interactive multi-party

Like non-interactive multi-party, each client is assigned `user_id`. After learning their `id`, clients participate in a 2 round protocol. In round 1, clients generate public key shares, share it with each other (either with or without server), and then aggregate the public key shares to produce collective public key. In round 2, clients, using collective public key, generate their server key shares and, along with encryptions of private inputs, upload it to the server. After which server can evaluate any arbitrary function on clients private inputs received along with server key shares or anytime in the future.

### Multi-party decryption

To decrypt encrypted outputs obtained as result of some computation, the clients need to come online. They download output ciphertext(s) from the server and generate decryption shares. Clients communicate their decryption shares with each other (with or without server). After receiving decryption shares from each of the other client, the clients can independently decrypt the output ciphertext(s).

### Parameter selection

We provide parameters to run both multi-party protocols for upto 8 parties.

| <= # Parties | Interactive multi-party | Non-interactive multi-party |
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

In encrypted domain there's no way to panic upon division by zero at runtime. Instead we set a local flag ciphertext accessible via `div_by_zero_flag()` that stores a boolean ciphertext indicating whether any of the divisions performed during the execution attempted division by zero. Assuming division by zero detection will be critical, we recommend decrypting the flag ciphertext along with other output ciphertexts in multi-party decryption procedure.

The div by zero flag is thread local. If you run multiple different FHE circuits in sequence without stopping the thread (i.e. within a single program) you will have to reset div by zero error flag before starting of the next in-sequence circuit execution with `reset_error_flags()`.

Please refer to [div_by_zero](./examples/div_by_zero.rs) example for more details.

**If and else using mux**

Branching in encrypted domain is expensive because the code must execute all the branches. Hence cost grows exponentially with no. of conditional branches. In general we recommend to modify the code to minimise conditional branches. However, if a code cannot be modified to made branchless, we provide `mux` API for FheUint8s. `mux` selects one of the two FheUint8s based on a selector bit. Please refer to [if_and_else](./examples/if_and_else.rs) example for more details.

### Security

> [!WARNING]
> Code has not been audited and, at the moment, we don't provide any security guarantees. We don't recommend to deploy it in production or to be used to handle important data.

All provided parameters are $2^{128}$ ring operations secure and have failure probability of at-least $2^{-40}$. However, there are two vital points to keep in mind:

1. Any user, without refreshing secrets, must not generate decryption shares for a given ciphertext more than once. Technically, it is insecure if a secret key generates two different decryption shares for the same ciphertext because it may lead to recovery of the ideal secret key with certain probability. At the moment, we recommend the users to maintain a local table that tracks ciphertexts for which they have generated decryption shares using their secret. And only generate a new decryption share for a ciphertext if the ciphertext does not exists in the local table.
2. At the moment, users, without refreshing secrets, must not run the protocol twice using the same application seed and produce different outputs. Technically, it is insecure if two different MPC transcripts are produced using same application seed and user secret. However, this must be handled within the library and is a pending feature to be implemented in future.
