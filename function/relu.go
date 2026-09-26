package function

import (
	"github.com/itsubaki/autograd/math"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// ReLU applies the rectified linear unit function.
func ReLU(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &ReLUT{},
	}).First(x...)
}

// ReLUT is the differentiable ReLU operation.
type ReLUT struct {
	x *variable.Variable
}

func (f *ReLUT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]

	y := tensor.F(x[0].Data, maximum)
	return []*variable.Variable{
		variable.From(y),
	}
}

func (f *ReLUT) Backward(gy ...*variable.Variable) []*variable.Variable {
	mask := tensor.Mask(f.x.Data, relu)
	return []*variable.Variable{
		Mul(gy[0], variable.From(mask)), // gy * mask
	}
}

func maximum(v float32) float32 { return math.Max(v, 0.0) }

func relu(v float32) bool { return v > 0 }
