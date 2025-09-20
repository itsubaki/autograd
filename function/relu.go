package function

import (
	"math"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func ReLU(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &ReLUT{}}).First(x...)
}

type ReLUT struct {
	x *variable.Variable
}

func (f *ReLUT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]

	return []*variable.Variable{
		variable.NewFrom(tensor.F(x[0].Data, maximum)),
	}
}

func (f *ReLUT) Backward(gy ...*variable.Variable) []*variable.Variable {
	mask := tensor.Mask(f.x.Data, relu)
	return []*variable.Variable{
		Mul(gy[0], variable.NewFrom(mask)), // gy * mask
	}
}

func maximum(v float64) float64 { return math.Max(v, 0.0) }

func relu(v float64) bool { return v > 0 }
