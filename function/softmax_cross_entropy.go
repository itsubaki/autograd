package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// SoftmaxCrossEntropy computes the softmax cross-entropy loss.
// It expects `x[0]` to have shape (N, C) and `x[1]`(`t`) to have shape (N,).
func SoftmaxCrossEntropy(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &SoftmaxCrossEntropyT{},
	}).First(x...)
}

type SoftmaxCrossEntropyT struct {
	N, C  int
	x     *variable.Variable
	label []int
}

func (f *SoftmaxCrossEntropyT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]
	f.N, f.C = x[0].Shape()[0], x[0].Shape()[1] // (N, C)
	f.label = tensor.Int(x[1].Data).Data        // (N,)

	logz := logsumexp(x[0].Data)                       // (N, 1)
	logp := logp(tensor.Sub(x[0].Data, logz), f.label) // (N, 1)
	sum := tensor.Sum(logp).At()                       // scalar

	return []*variable.Variable{
		variable.New(-1.0 / float64(f.N) * sum),
	}
}

func (f *SoftmaxCrossEntropyT) Backward(gy ...*variable.Variable) []*variable.Variable {
	t := variable.From(oneHot(f.label, f.C))
	y := Softmax(1)(f.x)
	yt := MulC(1.0/float64(f.N), Sub(y, t)) // (y - t)/N
	gx := Mul(yt, gy[0])                    // (y - t)/N * gy

	return []*variable.Variable{
		Reshape(f.x.Shape()...)(gx),
	}
}

// oneHot converts a slice of integer labels into a one-hot encoded tensor.
func oneHot(label []int, CNums int) *tensor.Tensor[float64] {
	out := tensor.Zeros[float64](len(label), CNums)
	for i, v := range label {
		out.Set([]int{i, v}, 1.0)
	}

	return out
}

// logsumexp computes the log of the sum of exponentials of the input tensor x.
func logsumexp(x *tensor.Tensor[float64]) *tensor.Tensor[float64] {
	// log(sum(exp(x))) = m + log(sum(exp(x - max)))
	max1 := tensor.Expand(tensor.Max(x, 1), 1)    // max1 = max(x, axis=1)
	expy := tensor.Exp(tensor.Sub(x, max1))       // expy = exp(x - max1)
	sum1 := tensor.Expand(tensor.Sum(expy, 1), 1) // sum1 = sum(expy)
	logsum1 := tensor.Log(sum1)                   // logsum1 = log(sum1)
	return tensor.Add(max1, logsum1)              // logsumexp = max1 + logsum1
}

// logp extracts the values from x corresponding to the true labels.
func logp(x *tensor.Tensor[float64], label []int) *tensor.Tensor[float64] {
	out := tensor.Zeros[float64](len(label), 1)
	for i, v := range label {
		out.Set([]int{i, 0}, x.At(i, v))
	}

	return out
}
