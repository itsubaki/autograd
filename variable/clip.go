package variable

import "github.com/itsubaki/autograd/tensor"

// Clip returns a function that clips x[0] to the interval [min, max].
func Clip(min, max float32) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &ClipT{
			Min: min,
			Max: max,
		},
	}).First
}

// ClipT is the differentiable clipping operation.
type ClipT struct {
	Min, Max float32
	x        *Variable
}

func (f *ClipT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := tensor.Clip(x[0].Data, f.Min, f.Max)
	return []*Variable{
		From(y),
	}
}

func (f *ClipT) Backward(gy ...*Variable) []*Variable {
	mask := tensor.Mask(f.x.Data, clip(f.Min, f.Max))
	return []*Variable{
		Mul(gy[0], From(mask)), // gy * mask
	}
}

// clip returns a function that checks if a value v is within the interval [min, max].
func clip(min, max float32) func(v float32) bool {
	return func(v float32) bool {
		return min < v && v < max
	}
}
