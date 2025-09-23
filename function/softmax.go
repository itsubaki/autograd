package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func Softmax(axis int) func(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &SoftmaxT{
			Axis: axis,
		},
	}).First
}

type SoftmaxT struct {
	Axis int
	y    *variable.Variable
}

func (f *SoftmaxT) Forward(x ...*variable.Variable) []*variable.Variable {
	max1 := tensor.Expand(tensor.Max(x[0].Data, f.Axis), f.Axis) // max1 = max(x, axis=1)
	expy := tensor.Exp(tensor.Sub(x[0].Data, max1))              // expy = exp(x - max1)
	sum1 := tensor.Expand(tensor.Sum(expy, f.Axis), f.Axis)      // sum1 = sum(expy, axis=1)
	div := tensor.Div(expy, sum1)                                // y = expy / sum1

	f.y = variable.From(div)
	return []*variable.Variable{
		f.y,
	}
}

func (f *SoftmaxT) Backward(gy ...*variable.Variable) []*variable.Variable {
	shape := keepDims(f.y.Shape(), f.Axis)

	gyy := Mul(gy[0], f.y)        // gyy = gy * y
	sum := SumTo(shape...)(gyy)   // sum = sum(gy, axis=1)
	gx := Sub(gyy, Mul(f.y, sum)) // gyy - y * sum
	return []*variable.Variable{
		gx,
	}
}

func keepDims(shape []int, axis int) []int {
	ndim := len(shape)
	if axis < 0 {
		axis += ndim
	}

	shape[axis] = 1
	return shape
}
