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
	label := vector.Int(matrix.Flatten(x[1].Data))

	logz := logsumexp(x[0].Data)
	logp := get(matrix.Sub(x[0].Data, logz), label)
	y := -1.0 / float64(N) * matrix.Sum(logp)

	return []*variable.Variable{
		variable.New(y),
	}
}

func (f *SoftmaxCrossEntropyT) Backward(gy ...*variable.Variable) []*variable.Variable {
	xs := f.x.Shape()
	N, C := xs[0], xs[1]

	t := variable.NewOf(onehot(vector.Int(f.t.Data[0]), C)...) // t = onehot(t, C)
	y := Softmax(f.x)                                          // y = softmax(x)
	gx := Mul(Sub(y, t), MulC(1.0/float64(N), gy[0]))          // (y - t) * gy / N

	return []*variable.Variable{
		gx,
	}
}

func logsumexp(x [][]float64) [][]float64 {
	max := matrix.BroadcastTo(matrix.Shape(x), matrix.MaxAxis1(x))        // max = max(x)
	expy := matrix.Exp(matrix.Sub(x, max))                                // expy = exp(x - max)
	sumy := matrix.BroadcastTo(matrix.Shape(expy), matrix.SumAxis1(expy)) // sumy = sum(expy)
	logsumy := matrix.Log(sumy)                                           // logsumy = log(sumy)
	return matrix.Add(max, logsumy)                                       // logsumexp = max + logsumy
}

func get(logp [][]float64, label []int) [][]float64 {
	logpt := make([][]float64, len(label))
	for i, v := range label {
		logpt[i] = []float64{logp[i][v]}
	}

	return logpt
}

func onehot(x []int, size int) [][]float64 {
	out := matrix.Zero(len(x), size)
	for i, v := range x {
		out[i][v] = 1
	}

	return out
}
