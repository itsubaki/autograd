package function

import "github.com/itsubaki/autograd/variable"

var (
	// primitive functions
	AddC        = variable.AddC
	Add         = variable.Add
	SubC        = variable.SubC
	Sub         = variable.Sub
	MulC        = variable.MulC
	Mul         = variable.Mul
	Div         = variable.Div
	DivC        = variable.DivC
	Sin         = variable.Sin
	Cos         = variable.Cos
	Tanh        = variable.Tanh
	Exp         = variable.Exp
	Log         = variable.Log
	Pow         = variable.Pow
	Square      = variable.Square
	Neg         = variable.Neg
	Sum         = variable.Sum
	SumTo       = variable.SumTo
	BroadcastTo = variable.BroadcastTo
	Reshape     = variable.Reshape
	Transpose   = variable.Transpose
	MatMul      = variable.MatMul
)
