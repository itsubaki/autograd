package numerical

import (
	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

type Func func(x ...*variable.Variable) *variable.Variable

var (
	_ Func = F.Add
	_ Func = F.Sub
	_ Func = F.Mul
	_ Func = F.Div
	_ Func = F.Sin
	_ Func = F.Cos
	_ Func = F.Tanh
	_ Func = F.Exp
	_ Func = F.Log
	_ Func = F.Pow(2.0)
	_ Func = F.Square
	_ Func = F.Neg
	_ Func = F.Sum
	_ Func = F.SumTo(1, 1)
	_ Func = F.BroadcastTo(1, 3)
	_ Func = F.Reshape(2, 2)
	_ Func = F.Transpose
	_ Func = F.MatMul
	_ Func = F.Max
	_ Func = F.Min
	_ Func = F.Clip(0.0, 1.0)
	_ Func = F.Linear
	_ Func = F.Sigmoid
	_ Func = F.ReLU
	_ Func = F.MeanSquaredError
	_ Func = F.GetItem([]int{0, 0, 1})
	_ Func = F.GetItemGrad([]int{0, 0, 1}, []int{2, 3})
	_ Func = F.Softmax
	_ Func = F.SoftmaxCrossEntropy
	_ Func = F.Dropout(0.5)
)

func Diff(f Func, x []*variable.Variable, h ...float64) *variable.Variable {
	if len(h) == 0 {
		h = append(h, 1e-4)
	}

	y0 := f(xh(x, h[0], matrix.AddC)...)          // f(x+h)
	y1 := f(xh(x, -1.0*h[0], matrix.AddC)...)     // f(x-h)
	df := matrix.F2(y0.Data, y1.Data, diff(h[0])) // (f(x+h) - f(x-h)) / 2h
	return &variable.Variable{Data: df}
}

func diff(h float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return (a - b) / (2 * h) }
}

func xh(x []*variable.Variable, h float64, f func(c float64, v matrix.Matrix) matrix.Matrix) []*variable.Variable {
	x0 := make([]*variable.Variable, len(x))
	for i := range x {
		x0[i] = variable.NewOf(f(h, x[i].Data)...)
	}

	return x0
}
