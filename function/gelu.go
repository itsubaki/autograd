package function

import (
	"math"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

const (
	c           = 0.044715
	sqrt2overPi = 0.7978845608028654
)

// GELU applies the Gaussian Error Linear Unit function.
func GELU(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &GELUT{},
	}).First(x...)
}

// GELUT is the differentiable GELU operation.
type GELUT struct {
	x *variable.Variable
}

func (f *GELUT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]

	y := tensor.F(x[0].Data, func(x float64) float64 {
		return 0.5 * x * (1.0 + math.Tanh(sqrt2overPi*(x+c*x*x*x)))
	})

	return []*variable.Variable{
		variable.From(y),
	}
}

func (f *GELUT) Backward(gy ...*variable.Variable) []*variable.Variable {
	x2, x3 := Pow(2)(f.x), Pow(3)(f.x)
	tanh := Tanh(MulC(sqrt2overPi, Add(f.x, MulC(c, x3))))
	a := AddC(0.5, MulC(0.5, tanh))
	b := MulC(0.5, Mul(f.x, SubC(1.0, Pow(2)(tanh))))
	du := MulC(sqrt2overPi, AddC(1.0, MulC(3.0*c, x2)))
	return []*variable.Variable{
		Mul(gy[0], Add(a, Mul(b, du))),
	}
}
