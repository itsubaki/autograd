package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// Minimum returns a function that computes the element-wise minimum of two variables.
func Minimum(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &MinimumT{},
	}).First(x...)
}

type MinimumT struct {
	x0Shape, x1Shape []int
	mask             *tensor.Tensor[float32]
}

func (f *MinimumT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x0Shape, f.x1Shape = x[0].Shape(), x[1].Shape()

	y, mask := tensor.Minimum[float32, float32](x[0].Data, x[1].Data)
	f.mask = mask

	return []*variable.Variable{
		variable.From(y),
	}
}

func (f *MinimumT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gx0 := tensor.SumTo(tensor.Mul(gy[0].Data, f.mask), f.x0Shape...)
	gx1 := tensor.SumTo(tensor.Mul(gy[0].Data, tensor.SubC(1, f.mask)), f.x1Shape...)

	return []*variable.Variable{
		variable.From(gx0),
		variable.From(gx1),
	}
}
