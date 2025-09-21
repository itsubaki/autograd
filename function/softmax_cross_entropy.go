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
	x, t *variable.Variable
}

func (f *SoftmaxCrossEntropyT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x, f.t = x[0], x[1]

	label := label(x[1])
	logz := logsumexp(x[0].Data)
	logp := logp(tensor.Sub(x[0].Data, logz), label)
	N := x[0].Shape()[0]

	y := -1.0 / float64(N) * tensor.Sum(logp).At()
	return []*variable.Variable{
		variable.New(y),
	}
}

func (f *SoftmaxCrossEntropyT) Backward(gy ...*variable.Variable) []*variable.Variable {
	shape := f.x.Shape()
	N, C := shape[0], shape[1]

	t := variable.From(onehot(f.t.Data.Data, C)) // t = onehot(t, C)
	y := Softmax(f.x)                            // y = softmax(x)
	return []*variable.Variable{
		Mul(Sub(y, t), MulC(1.0/float64(N), gy[0])), // (y - t) * gy / N
	}
}

func logsumexp(x *tensor.Tensor[float64]) *tensor.Tensor[float64] {
	max := tensor.Expand(tensor.Max(x, 1), 1)
	expy := tensor.Exp(tensor.Sub(x, max))        // expy = exp(x - max)
	sumy := tensor.Expand(tensor.Sum(expy, 1), 1) // sumy = sum(expy)
	logsumy := tensor.Log(sumy)                   // logsumy = log(sumy)
	return tensor.Add(max, logsumy)               // logsumexp = max + logsumy
}

func label(t *variable.Variable) []int {
	return toInt(t.Data.Data)
}

func logp(m *tensor.Tensor[float64], label []int) *tensor.Tensor[float64] {
	out := tensor.Zeros[float64](len(label), 1)
	for i, v := range label {
		out.Set([]int{i, 0}, m.At(i, v))
	}

	return out
}

func onehot(t []float64, size int) *tensor.Tensor[float64] {
	x := toInt(t)
	out := tensor.Zeros[float64](len(x), size)

	for i, v := range x {
		out.Set([]int{i, v}, 1)
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
