Noninteractive MP-FHE Setup
(1) Client keys generated, 0ms
(2) Clients generate server key shares, 5310ms
(3) Server key aggregated, 40360ms

F1 computation
(1) Clients encrypt their input with their own key, 0ms
(2) Client inputs extracted after key switch, 5ms
(3) f1 evaluated, 23977ms
(4) Decryption shares generated, 0ms
(5) Decryption shares aggregated, data decrypted by client, 0ms

F1 computation
(1) Clients encrypt their input with their own key, 0ms
(2) Client inputs extracted after key switch, 6ms
(3) f2 evaluated, 23793ms
(4) Decryption shares generated, 0ms
(5) Decryption shares aggregated, data decrypted by client, 0ms

---

Interactive MP-FHE Setup
(1) Client keys generated, 0ms
(2) Public key shares generated, 1ms
(3) Aggregated public key shares generated, 0ms
(4) Server key shares generated, 3700ms
(5) Server key aggregated from shares, 2354ms

F1 computation
(1) All client inputs encrypted with collective public key, 0ms
(2) Client inputs extracted from batched ciphertext, 0ms
(3) f1 evaluated, 24428ms
(4) Decryption shares generated, 0ms
(5) Decryption shares aggregated, data decrypted by client, 0ms

F2 computation
(1) All client inputs encrypted with collective public key, 0ms
(2) Client inputs extracted from batched ciphertext, 0ms
(3) f2 evaluated, 24173ms
(4) Decryption shares generated, 0ms
(5) Decryption shares aggregated, data decrypted by client, 0ms