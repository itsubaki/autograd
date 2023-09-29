package function

import "github.com/itsubaki/autograd/variable"

var (
	AddC        = variable.AddC
	Add         = variable.Add
	SubC        = variable.SubC
	Sub         = variable.Sub
	MulC        = variable.MulC
	Mul         = variable.Mul
	Div         = variable.Div
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
)
