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
	mask *tensor.Tensor[float64]
}

func (f *MinimumT) Forward(x ...*variable.Variable) []*variable.Variable {
	y, mask := tensor.Minimum[float64, float64](x[0].Data, x[1].Data)
	f.mask = mask

	return []*variable.Variable{
		variable.From(y),
	}
}

func (f *MinimumT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gx0 := tensor.Mul(gy[0].Data, f.mask)
	gx1 := tensor.Mul(gy[0].Data, tensor.SubC(1, f.mask))

	return []*variable.Variable{
		variable.From(gx0),
		variable.From(gx1),
	}
}
