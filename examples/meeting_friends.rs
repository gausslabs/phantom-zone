use bin_rs::*;
use itertools::Itertools;
use rand::{thread_rng, RngCore};

struct Location<T>(T, T);

impl<T> Location<T> {
    fn new(x: T, y: T) -> Self {
        Location(x, y)
    }

    fn x(&self) -> &T {
        &self.0
    }
    fn y(&self) -> &T {
        &self.1
    }
}

fn should_meet(a: &Location<u8>, b: &Location<u8>, b_threshold: &u8) -> bool {
    let diff_x = a.x() - b.x();
    let diff_y = a.y() - b.y();
    let d_sq = &(&diff_x * &diff_x) + &(&diff_y * &diff_y);

    d_sq.le(b_threshold)
}

/// Calculates distance square between a's and b's location. Returns a boolean
/// indicating whether diatance sqaure is <= `b_threshold`.
fn should_meet_fhe(
    a: &Location<FheUint8>,
    b: &Location<FheUint8>,
    b_threshold: &FheUint8,
) -> FheBool {
    let diff_x = a.x() - b.x();
    let diff_y = a.y() - b.y();
    let d_sq = &(&diff_x * &diff_x) + &(&diff_y * &diff_y);

    d_sq.le(b_threshold)
}

// Even wondered who are the long distance friends (friends of friends or
// friends of friends of friends...) that live nearby ? But how do you find
// them? Surely no-one will simply reveal their exact location just because
// there's a slight chance that a long distance friend lives nearby.
//
// Here we write a simple application with two users `a` and `b`. User `a` wants
// to find (long distance) friends that live in their neighbourhood. User `b` is
// open to meeting new friends within some distance of their location. Both user
// `a` and `b` encrypt their location and upload to the server. User `b` also
// encrypts the distance square threshold within which they are interested in
// meeting new friends. The server calculates the square of the distance between
// user a's location and user b's location and returns encrypted boolean output
// indicating whether square of distance is <= user b's supplied distance square
// threshold. User `a` then comes online, downloads output ciphertext, produces
// their decryption share for user `b`, and uploads the decryption share to the
// server. User `b` comes online, downloads output ciphertext and user a's
// decryption share, produces their own decryption share, and then decrypts the
// encrypted boolean output. If the output is `True`, it indicates
// user `a` is within the distance square threshold defined by user `b`.
fn main() {
    set_parameter_set(ParameterSelector::NonInteractiveLTE2Party);

    // set application's common reference seed
    let mut seed = [0u8; 32];
    thread_rng().fill_bytes(&mut seed);
    set_common_reference_seed(seed);

    let no_of_parties = 2;

    // Client Side //

    // Generate client keys
    let cks = (0..no_of_parties).map(|_| gen_client_key()).collect_vec();

    // We assign id 0 to client 0 and id 1 to client 1
    let a_id = 0;
    let b_id = 1;
    let user_a_secret = &cks[0];
    let user_b_secret = &cks[1];

    // User a and b generate server key shares
    let a_server_key_share = gen_server_key_share(a_id, no_of_parties, user_a_secret);
    let b_server_key_share = gen_server_key_share(b_id, no_of_parties, user_b_secret);

    // User a and b encrypt their locations
    let user_a_secret = &cks[0];
    let user_a_location = Location::new(50, 60);
    let user_a_enc =
        user_a_secret.encrypt(vec![*user_a_location.x(), *user_a_location.y()].as_slice());

    let user_b_location = Location::new(50, 60);
    // User b also encrypts the distance sq threshold
    let user_b_threshold = 20;
    let user_b_enc = user_b_secret
        .encrypt(vec![*user_b_location.x(), *user_b_location.y(), user_b_threshold].as_slice());

    // Server Side //

    // Both user a and b upload their private inputs and server key shares to
    // the server in one shot message
    let server_key = aggregate_server_key_shares(&vec![a_server_key_share, b_server_key_share]);
    server_key.set_server_key();

    // Server parses private inputs from user a and b
    let user_a_location_enc = {
        let c = user_a_enc.unseed::<Vec<Vec<u64>>>().key_switch(0);
        Location::new(c.extract(0), c.extract(1))
    };
    let (user_b_location_enc, user_b_threshold_enc) = {
        let c = user_b_enc.unseed::<Vec<Vec<u64>>>().key_switch(1);
        (Location::new(c.extract(0), c.extract(1)), c.extract(2))
    };

    // run the circuit
    let out_c = should_meet_fhe(
        &user_a_location_enc,
        &user_b_location_enc,
        &user_b_threshold_enc,
    );

    // Client Side //

    // user a comes online, downloads out_c, produces a decryption share, and
    // uploads the decryption share to the server.
    let a_dec_share = user_a_secret.gen_decryption_share(&out_c);

    // user b comes online downloads user a's decryption share, generates their
    // own decryption share, decrypts the output ciphertext. If the output is
    // True, they contact user a to meet.
    let b_dec_share = user_b_secret.gen_decryption_share(&out_c);
    let out_bool =
        user_b_secret.aggregate_decryption_shares(&out_c, &vec![b_dec_share, a_dec_share]);

    assert_eq!(
        out_bool,
        should_meet(&user_a_location, &user_b_location, &user_b_threshold)
    );

    if out_bool {
        println!("A lives nearby. B should meet A.");
    } else {
        println!("A lives too far away!")
    }
}
