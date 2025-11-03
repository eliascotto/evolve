extern crate rustyline;
extern crate itertools;
extern crate logos;

use evolve::repl::REPL;

fn main() {
    let repl = REPL {};
    repl.run();
}
