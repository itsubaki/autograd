package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// MaskFill returns a function that fills elements of x with the given value v where the corresponding elements of mask are 0.
// This is typically used for attention masking in Transformer models,
// e.g. filling masked positions with a large negative value before softmax.
func MaskFill(mask *tensor.Tensor[float64], f func(m float64) bool, v float64) func(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &MaskFillT{
			mask: mask,
			fill: v,
			cond: f,
		},
	}).First
}

type MaskFillT struct {
	mask *tensor.Tensor[float64]
	fill float64
	cond func(m float64) bool
}

func (f *MaskFillT) Forward(x ...*variable.Variable) []*variable.Variable {
	filled := tensor.MaskFill(x[0].Data, f.mask, f.cond, f.fill)

	return []*variable.Variable{
		variable.From(filled),
	}
}

func (f *MaskFillT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Mul(gy[0], variable.From(f.mask)),
	}
}
