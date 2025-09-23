package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// SoftmaxCrossEntropy computes the softmax cross-entropy loss.
// It expects `x[0]` to have shape (N, C, ...) and `x[1]`(`t`) to have shape (N, ...).
func SoftmaxCrossEntropy(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &SoftmaxCrossEntropyT{
			AxisN: 0,
			AxisC: 1,
		},
	}).First(x...)
}

type SoftmaxCrossEntropyT struct {
	AxisN, AxisC int
	N, C         int
	x            *variable.Variable
	label        []int
}

func (f *SoftmaxCrossEntropyT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]
	f.label = tensor.Int(tensor.Reshape(x[1].Data, flatten(x[1].Shape(), f.AxisC)...)).Data // (N, )
	xFlat := tensor.Reshape(x[0].Data, flatten(x[0].Shape(), f.AxisC)...)                   // (N, C)
	f.N, f.C = xFlat.Shape[f.AxisN], xFlat.Shape[f.AxisC]

	logz := logsumexp(xFlat)
	logp := logp(tensor.Sub(xFlat, logz), f.label)
	sum := tensor.Sum(logp).At()

	return []*variable.Variable{
		variable.New(-1.0 / float64(f.N) * sum),
	}
}

func (f *SoftmaxCrossEntropyT) Backward(gy ...*variable.Variable) []*variable.Variable {
	t := variable.From(oneHot(f.label, f.C))                              // t
	y := Softmax(f.AxisC)(Reshape(flatten(f.x.Shape(), f.AxisC)...)(f.x)) // y
	yt := MulC(1.0/float64(f.N), Sub(y, t))                               // (y - t)/N
	gx := Mul(yt, gy[0])                                                  // (y - t)/N * gy

	return []*variable.Variable{
		Reshape(f.x.Shape()...)(gx),
	}
}

func oneHot(label []int, CNums int) *tensor.Tensor[float64] {
	out := tensor.Zeros[float64](len(label), CNums)
	for i, v := range label {
		out.Set([]int{i, v}, 1.0)
	}

	return out
}

func logsumexp(x *tensor.Tensor[float64]) *tensor.Tensor[float64] {
	// log(sum(exp(x))) = m + log(sum(exp(x - max)))
	max1 := tensor.Expand(tensor.Max(x, 1), 1)    // max1 = max(x, axis=1)
	expy := tensor.Exp(tensor.Sub(x, max1))       // expy = exp(x - max1)
	sum1 := tensor.Expand(tensor.Sum(expy, 1), 1) // sum1 = sum(expy)
	logsum1 := tensor.Log(sum1)                   // logsum1 = log(sum1)
	return tensor.Add(max1, logsum1)              // logsumexp = max1 + logsum1
}

func logp(x *tensor.Tensor[float64], label []int) *tensor.Tensor[float64] {
	out := tensor.Zeros[float64](len(label), 1)
	for i, v := range label {
		out.Set([]int{i, 0}, x.At(i, v))
	}

	return out
}

// flatten reshapes the input shape to (N, C) where C is the dimension of the given axis.
func flatten(shape []int, axis int) []int {
	N := 1
	for i, s := range shape {
		if i == axis {
			continue
		}

		N *= s
	}

	return []int{N, shape[axis]} // (N, C)
}
