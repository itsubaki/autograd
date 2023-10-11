package function

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func SoftmaxCrossEntropy(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &SoftmaxCrossEntropyT{}}).ApplyAndFirst(x...)
}

type SoftmaxCrossEntropyT struct {
	x, t *variable.Variable
}

func (f *SoftmaxCrossEntropyT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x, f.t = x[0], x[1]

	N := len(x[0].Data)
	label := label(x[1])
	logz := logsumexp(x[0].Data)
	logp := logp(matrix.Sub(x[0].Data, logz), label)

	y := -1.0 / float64(N) * matrix.Sum(logp)
	return []*variable.Variable{
		variable.New(y),
	}
}

func (f *SoftmaxCrossEntropyT) Backward(gy ...*variable.Variable) []*variable.Variable {
	shape := variable.Shape(f.x)
	N, C := shape[0], shape[1]

	t := variable.NewOf(onehot(f.t.Data[0], C)...) // t = onehot(t, C)
	y := Softmax(f.x)                              // y = softmax(x)

	return []*variable.Variable{
		Mul(Sub(y, t), MulC(1.0/float64(N), gy[0])), // (y - t) * gy / N
	}
}

func logsumexp(x [][]float64) [][]float64 {
	max := matrix.MaxAxis1(x)              // max = max(x)
	expy := matrix.Exp(matrix.Sub(x, max)) // expy = exp(x - max)
	sumy := matrix.SumAxis1(expy)          // sumy = sum(expy)
	logsumy := matrix.Log(sumy)            // logsumy = log(sumy)
	return matrix.Add(max, logsumy)        // logsumexp = max + logsumy
}

func label(t *variable.Variable) []int {
	return vector.Int(matrix.Flatten(t.Data))
}

func logp(m [][]float64, label []int) [][]float64 {
	out := make([][]float64, len(label))
	for i, v := range label {
		out[i] = []float64{m[i][v]}
	}

	return out
}

func onehot(t []float64, size int) [][]float64 {
	x := vector.Int(t)

	out := matrix.Zero(len(x), size)
	for i, v := range x {
		out[i][v] = 1
	}

	return out
}
