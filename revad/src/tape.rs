use std::cell::RefCell;

#[derive(Clone, Copy)]
struct Node {
    weights: [f64; 2],
    deps: [usize; 2],
}

pub struct Tape { nodes: RefCell<Vec<Node>> }

impl Tape {
    pub fn new() -> Self {
        Tape { nodes: RefCell::new(Vec::new()) }
    }

    pub fn var(&self, value: f64) -> Var<'_> {
        Var {
            tape: self,
            value,
            index: self.push0(),
        }
    }

    fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    fn push0(&self) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node {
            weights: [0.0, 0.0],
            deps: [len, len],
        });
        len
    }

    fn push1(&self, dep0: usize, weight0: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node {
            weights: [weight0, 0.0],
            deps: [dep0, len],
        });
        len
    }

    fn push2(&self,
             dep0: usize, weight0: f64,
             dep1: usize, weight1: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node {
            weights: [weight0, weight1],
            deps: [dep0, dep1],
        });
        len
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
pub struct Var<'t> {
    tape: &'t Tape,
    index: usize,
    value: f64,
}

impl<'t> Var<'t> {
    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn grad(&self) -> Grad {
        let len = self.tape.len();
        let nodes = self.tape.nodes.borrow();
        let mut derivs = vec![0.0; len];
        derivs[self.index] = 1.0;
        for i in (0 .. len).rev() {
            let node = nodes[i];
            let deriv = derivs[i];
            for j in 0 .. 2 {
                derivs[node.deps[j]] += node.weights[j] * deriv;
            }
        }
        Grad { derivs }
    }

    pub fn abs(self) -> Self {
        Var {
            tape: self.tape,
            value: self.value.abs(),
            index: self.tape.push1(
                self.index, if self.value.is_sign_negative() { -1.0 } else { 1.0 }
            ),
        }
    }

    pub fn sin(self) -> Self {
        Var {
            tape: self.tape,
            value: self.value.sin(),
            index: self.tape.push1(
                self.index, self.value.cos(),
            ),
        }
    }

    pub fn tanh(self) -> Self {
        Var {
            tape: self.tape,
            value: self.value.tanh(),
            index: self.tape.push1(
                self.index, 1.0 / self.value * self.value,
            ),
        }
    }
}

impl<'t> ::std::ops::Neg for Var<'t> {
    type Output = Var<'t>;
    fn neg(self) -> Self::Output {
        Var {
            tape: self.tape,
            value: -self.value,
            index: self.tape.push1(
                self.index, -1.0
            ),
        }
    }
}

impl<'t> ::std::ops::Add for Var<'t> {
    type Output = Var<'t>;
    fn add(self, other: Var<'t>) -> Self::Output {
        assert_eq!(self.tape as *const Tape, other.tape as *const Tape);
        Var {
            tape: self.tape,
            value: self.value + other.value,
            index: self.tape.push2(
                self.index, 1.0,
                other.index, 1.0,
           ),
        }
    }
}

impl<'t> ::std::ops::Sub for Var<'t> {
    type Output = Var<'t>;
    fn sub(self, other: Var<'t>) -> Self::Output {
        assert_eq!(self.tape as *const Tape, other.tape as *const Tape);
        Var {
            tape: self.tape,
            value: self.value - other.value,
            index: self.tape.push2(
                self.index, -1.0,
                other.index, -1.0,
           ),
        }
    }
}

impl<'t> ::std::ops::Mul for Var<'t> {
    type Output = Var<'t>;
    fn mul(self, other: Var<'t>) -> Self::Output {
        assert_eq!(self.tape as *const Tape, other.tape as *const Tape);
        Var {
            tape: self.tape,
            value: self.value * other.value,
            index: self.tape.push2(
                self.index, other.value,
                other.index, self.value,
            ),
        }
    }
}

impl<'t> ::std::ops::Div for Var<'t> {
    type Output = Var<'t>;
    fn div(self, other: Var<'t>) -> Self::Output {
        assert_eq!(self.tape as *const Tape, other.tape as *const Tape);
        Var {
            tape: self.tape,
            value: self.value / other.value,
            index: self.tape.push2(
                self.index, 1.0 / other.value,
                other.index, -self.value / (other.value * other.value),
            ),
        }
    }
}


pub struct Grad { derivs: Vec<f64> }

impl Grad {
    pub fn wrt(&self, var: Var<'_>) -> f64 {
        self.derivs[var.index]
    }
}

#[cfg(test)]
mod tests {
    use super::Tape;

    #[test]
    fn x_times_y_plus_sin_x() {
        let t = Tape::new();
        let x = t.var(0.5);
        let y = t.var(4.2);
        let z = x * y + x.sin();
        let grad = z.grad();
        assert!((z.value - 2.579425538604203).abs() <= 1e-15);
        assert!((grad.wrt(x) - (y.value + x.value.cos())).abs() <= 1e-15);
        assert!((grad.wrt(y) - x.value).abs() <= 1e-15);
    }
}
