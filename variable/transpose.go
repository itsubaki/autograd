package variable

import "github.com/itsubaki/autograd/tensor"

// TransposeMatMul returns a function that transposes a tensor by swapping
// the last two axes, which is often required when preparing inputs for MatMul.
//
// The permutation of axes is automatically constructed based on ndim.
// For example, if ndim = 4, the axes will be [0, 1, 3, 2], so the last
// two dimensions are reversed while the others remain in order.
func TransposeMatMul(ndim int) func(x ...*Variable) *Variable {
	axes := make([]int, ndim)
	for i := range ndim - 2 {
		axes[i] = i
	}

	// swap last two axes
	axes[ndim-2] = ndim - 1
	axes[ndim-1] = ndim - 2

	// transpose
	return Transpose(axes...)
}

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
	out := make([]int, ndim)
	for i, a := range axes {
		if a < 0 {
			a += ndim
		}

		out[a] = i
	}

	return out
}
