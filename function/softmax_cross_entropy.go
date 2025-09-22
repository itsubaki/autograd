package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func SoftmaxCrossEntropy(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &SoftmaxCrossEntropyT{},
	}).First(x...)
}

type SoftmaxCrossEntropyT struct {
	x     *variable.Variable
	label []int
}

func (f *SoftmaxCrossEntropyT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x, f.label = x[0], toInt(x[1].Data.Data)

	logz := logsumexp(x[0].Data)
	logp := logp(tensor.Sub(x[0].Data, logz), f.label)
	N := x[0].Shape()[0]

	y := -1.0 / float64(N) * tensor.Sum(logp).At()
	return []*variable.Variable{
		variable.New(y),
	}
}

func (f *SoftmaxCrossEntropyT) Backward(gy ...*variable.Variable) []*variable.Variable {
	y := Softmax(1)(f.x)
	for i, l := range f.label {
		y.Data.Set([]int{i, l}, y.Data.At(i, l)-1)
	}

	N := f.x.Shape()[0]
	gx := Mul(y, MulC(1.0/float64(N), gy[0])) // (y - t) * gy / N
	return []*variable.Variable{
		gx,
	}
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

func toInt(x []float64) []int {
	out := make([]int, len(x))
	for i, v := range x {
		out[i] = int(v)
	}

	return out
}
