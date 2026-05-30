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

	y := tensor.F(x[0].Data, gelu)
	return []*variable.Variable{
		variable.From(y),
	}
}

func (f *GELUT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gx := tensor.F(f.x.Data, geluGrad)
	return []*variable.Variable{
		Mul(gy[0], variable.From(gx)),
	}
}

func gelu(x float64) float64 {
	return 0.5 * x * (1.0 + math.Tanh(sqrt2overPi*(x+c*x*x*x)))
}

func geluGrad(x float64) float64 {
	tanh := math.Tanh(sqrt2overPi * (x + c*x*x*x))
	return 0.5*(1.0+tanh) + 0.5*x*(1.0-tanh*tanh)*(sqrt2overPi*(1.0+3.0*c*x*x))
}
