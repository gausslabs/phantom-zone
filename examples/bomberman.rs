use std::fmt::Debug;

use bin_rs::*;
use itertools::Itertools;
use rand::{thread_rng, Rng, RngCore};

struct Coordinates<T>(T, T);
impl<T> Coordinates<T> {
    fn new(x: T, y: T) -> Self {
        Coordinates(x, y)
    }
    fn x(&self) -> &T {
        &self.0
    }

    fn y(&self) -> &T {
        &self.1
    }
}

impl<T> Debug for Coordinates<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Coordinates")
            .field("x", self.x())
            .field("y", self.y())
            .finish()
    }
}

fn coordinates_is_equal(a: &Coordinates<FheUint8>, b: &Coordinates<FheUint8>) -> FheBool {
    &(a.x().eq(b.x())) & &(a.y().eq(b.y()))
}

fn traverse_map(p0: &[Coordinates<FheUint8>], bomb_coords: &[Coordinates<FheUint8>]) -> FheBool {
    // First move
    let mut out = coordinates_is_equal(&p0[0], &bomb_coords[0]);
    bomb_coords.iter().skip(1).for_each(|b_coord| {
        out |= coordinates_is_equal(&p0[0], &b_coord);
    });

    // rest of the moves
    p0.iter().skip(1).for_each(|m_coord| {
        bomb_coords.iter().for_each(|b_coord| {
            out |= coordinates_is_equal(m_coord, b_coord);
        });
    });

    out
}

// Do you recall bomberman? It's an interesting game where the bomberman has to
// cross the map without stepping on strategically placed bombs all across the
// map. Below we implement a very basic prototype of bomberman with 4 players.
//
// The map has 256 tiles with bottom left-most tile labelled with coordinate
// (0,0) and top right-most tile labelled with coordinate (255, 255). There are
// 4 players: Player 0, Player 1, Player 2, Player 3. Player 0's task is to walk
// across the map with fixed no. of moves while preventing itself from stepping
// on any of the bombs placed across the map by Player 1, 2, and 3.
//
// The twist is that Player 0's moves and the locations of bombs placed by other
// players are encrypted. Player 0 moves across the map in encrypted domain.
// Only a boolean output indicating whether player 0 survived after all the
// moves or killed itself by stepping onto a bomb is revealed at the end. If
// player 0 survives, Player 1, 2, 3 never learn what moves did it make. If
// player 0 kills itself by stepping onto a bomb, it only learns that bomb was
// placed on one coordinates it moved to. Moreover, Player 1, 2, 3 never learn
// about locations of each other bombs or even whose bomb killed Player 1.
fn main() {
    set_parameter_set(ParameterSelector::NonInteractiveLTE4Party);

    // set application's common reference seed
    let mut seed = [0; 32];
    thread_rng().fill_bytes(&mut seed);
    set_common_reference_seed(seed);

    let no_of_parties = 4;

    // Client side //

    let cks = (0..no_of_parties).map(|_| gen_client_key()).collect_vec();

    let server_key_shares = cks
        .iter()
        .enumerate()
        .map(|(index, k)| gen_server_key_share(index, no_of_parties, k))
        .collect_vec();

    // encrypt inputs
    let no_of_moves = 10;
    let player_0_moves = (0..no_of_moves)
        .map(|_| Coordinates::new(thread_rng().gen::<u8>(), thread_rng().gen()))
        .collect_vec();
    let player_1_bomb = Coordinates::new(thread_rng().gen::<u8>(), thread_rng().gen());
    let player_2_bomb = Coordinates::new(thread_rng().gen::<u8>(), thread_rng().gen());
    let player_3_bomb = Coordinates::new(thread_rng().gen::<u8>(), thread_rng().gen());

    println!("P0 move coordinates: {:?}", &player_0_moves);
    println!("P1 bomb coordinate : {:?}", &player_1_bomb);
    println!("P2 bomb coordinate : {:?}", &player_2_bomb);
    println!("P3 bomb coordinate : {:?}", &player_3_bomb);

    // Al playes encrypt their private inputs
    let player_0_enc = cks[0].encrypt(
        player_0_moves
            .iter()
            .flat_map(|c| vec![*c.x(), *c.y()])
            .collect_vec()
            .as_slice(),
    );
    let player_1_enc = cks[1].encrypt(vec![*player_1_bomb.x(), *player_1_bomb.y()].as_slice());
    let player_2_enc = cks[2].encrypt(vec![*player_2_bomb.x(), *player_2_bomb.y()].as_slice());
    let player_3_enc = cks[3].encrypt(vec![*player_3_bomb.x(), *player_3_bomb.y()].as_slice());

    // All player upload the encrypted inputs and server key shates to the server

    // Server side //

    let server_key = aggregate_server_key_shares(&server_key_shares);
    server_key.set_server_key();

    // server parses all player inputs
    let player_0_moves_enc = {
        let c = player_0_enc.unseed::<Vec<Vec<u64>>>().key_switch(0);
        (0..no_of_moves)
            .map(|i| Coordinates::new(c.extract(2 * i), c.extract(i * 2 + 1)))
            .collect_vec()
    };
    let player_1_bomb_enc = {
        let c = player_1_enc.unseed::<Vec<Vec<u64>>>().key_switch(1);
        Coordinates::new(c.extract(0), c.extract(1))
    };
    let player_2_bomb_enc = {
        let c = player_2_enc.unseed::<Vec<Vec<u64>>>().key_switch(2);
        Coordinates::new(c.extract(0), c.extract(1))
    };
    let player_3_bomb_enc = {
        let c = player_3_enc.unseed::<Vec<Vec<u64>>>().key_switch(3);
        Coordinates::new(c.extract(0), c.extract(1))
    };

    // run the game
    let player_0_dead_ct = traverse_map(
        &player_0_moves_enc,
        &vec![player_1_bomb_enc, player_2_bomb_enc, player_3_bomb_enc],
    );

    // All players generate decryption shares
    let decryption_shares = cks
        .iter()
        .map(|k| k.gen_decryption_share(&player_0_dead_ct))
        .collect_vec();
    let player_0_dead = cks[0].aggregate_decryption_shares(&player_0_dead_ct, &decryption_shares);

    if player_0_dead {
        println!("Oops! Player 0 dead");
    } else {
        println!("Wohoo! Player 0 survived");
    }
}
