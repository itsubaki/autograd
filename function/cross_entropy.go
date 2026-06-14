package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// CrossEntropy computes the softmax cross-entropy loss.
// It expects x[0] to have shape (N, C) and x[1] (t) to have shape (N,).
func CrossEntropy(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &CrossEntropyT{
			ignoreIndex: -100,
		},
	}).First(x...)
}

// CrossEntropyT is the differentiable softmax cross-entropy operation.
type CrossEntropyT struct {
	N, C        int
	x           *variable.Variable
	ignoreIndex int
	label       []int
}

func (f *CrossEntropyT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x, f.C = x[0], x[0].Shape()[1]     // (N, C)
	f.label = tensor.Int(x[1].Data).Data // (N,)
	f.N = count(f.label, f.ignoreIndex)
	if f.N == 0 {
		return []*variable.Variable{
			variable.New(0.0),
		}
	}

	logz := logsumexp(x[0].Data)                                      // (N, 1)
	logp := logp(tensor.Sub(x[0].Data, logz), f.label, f.ignoreIndex) // (N, 1)
	sum := tensor.Sum(logp).At()                                      // scalar

	return []*variable.Variable{
		variable.New(-1.0 / float64(f.N) * sum),
	}
}

func (f *CrossEntropyT) Backward(gy ...*variable.Variable) []*variable.Variable {
	if f.N == 0 {
		return []*variable.Variable{
			variable.Zeros(f.x.Shape()...),
		}
	}

	t := variable.From(oneHot(f.label, f.C, f.ignoreIndex)) // (N, C)
	y := Softmax(1)(f.x)                                    // (N, C)
	mask := ignoreMask(f.label, f.C, f.ignoreIndex)         // (N, C)
	diff := Mul(Sub(y, t), mask)                            // (y-t) * mask
	yt := MulC(1.0/float64(f.N), diff)                      // (y-t) * mask/N
	gx := Mul(yt, gy[0])                                    // (y-t) * mask/N * gy
	return []*variable.Variable{
		Reshape(f.x.Shape()...)(gx),
	}
}

// oneHot converts a slice of integer labels into a one-hot encoded tensor.
func oneHot(label []int, CNums int, ignoreIndex int) *tensor.Tensor[float64] {
	out := tensor.Zeros[float64](len(label), CNums)
	for i, v := range label {
		if v == ignoreIndex {
			continue
		}

		out.Set([]int{i, v}, 1.0)
	}

	return out
}

// logsumexp computes log(sum(exp(x))) for the input tensor x.
func logsumexp(x *tensor.Tensor[float64]) *tensor.Tensor[float64] {
	// log(sum(exp(x))) = m + log(sum(exp(x - max)))
	max1 := tensor.Expand(tensor.Max(x, 1), 1)    // max1 = max(x, axis=1)
	expy := tensor.Exp(tensor.Sub(x, max1))       // expy = exp(x - max1)
	sum1 := tensor.Expand(tensor.Sum(expy, 1), 1) // sum1 = sum(expy)
	logsum1 := tensor.Log(sum1)                   // logsum1 = log(sum1)
	return tensor.Add(max1, logsum1)              // logsumexp = max1 + logsum1
}

// logp extracts the values from x corresponding to the true labels.
func logp(x *tensor.Tensor[float64], label []int, ignoreIndex int) *tensor.Tensor[float64] {
	out := tensor.Zeros[float64](len(label), 1)
	for i, v := range label {
		if v == ignoreIndex {
			continue
		}

		out.Set([]int{i, 0}, x.At(i, v))
	}

	return out
}

func count(label []int, ignoreIndex int) int {
	var n int
	for _, v := range label {
		if v == ignoreIndex {
			continue
		}

		n++
	}

	return n
}

func ignoreMask(label []int, C, ignoreIndex int) *variable.Variable {
	mask := tensor.Ones[float64](len(label), C)
	for i, v := range label {
		if v != ignoreIndex {
			continue
		}

		for j := range C {
			mask.Set([]int{i, j}, 0.0)
		}
	}

	return variable.From(mask)
}
