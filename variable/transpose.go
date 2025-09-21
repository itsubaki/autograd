package variable

import (
	"github.com/itsubaki/autograd/tensor"
)

func Transpose(axes ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &TransposeT{
			Axes: axes,
		},
	}).First
}

type TransposeT struct {
	Axes []int
	ndim int
}

func (f *TransposeT) Forward(x ...*Variable) []*Variable {
	f.ndim = x[0].Data.NumDims()

	y := tensor.Transpose(x[0].Data, f.Axes...)
	return []*Variable{
		From(y),
	}
}

func (f *TransposeT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Transpose(invperm(f.ndim, f.Axes...)...)(gy[0]),
	}
}

func invperm(ndim int, axes ...int) []int {
	if ndim != len(axes) {
		panic("axes must specify all dimensions")
	}

	out := make([]int, ndim)
	for i, a := range axes {
		if a < 0 {
			a = ndim + a
		}

		if a < 0 || a >= ndim {
			panic("invalid axis index")
		}

		out[a] = i
	}

	return out
}
