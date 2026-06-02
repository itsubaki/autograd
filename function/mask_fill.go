package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// MaskFill returns a function that fills elements of x with the given value v where the corresponding elements of mask are 0.
// This is typically used for attention masking in Transformer models,
// e.g. filling masked positions with a large negative value before softmax.
func MaskFill(mask *tensor.Tensor[float64], v float64) func(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &MaskFillT{
			mask: mask,
			fill: v,
		},
	}).First
}

type MaskFillT struct {
	mask *tensor.Tensor[float64]
	fill float64
}

func (f *MaskFillT) Forward(x ...*variable.Variable) []*variable.Variable {
	filled := tensor.MaskFill(x[0].Data, f.mask, func(_, m float64) bool {
		return m == 0
	}, f.fill)

	return []*variable.Variable{
		variable.From(filled),
	}
}

func (f *MaskFillT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gx := tensor.Mul(gy[0].Data, f.mask)
	return []*variable.Variable{
		variable.From(gx),
	}
}
