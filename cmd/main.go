package main

import (
	"flag"
	"fmt"

	"github.com/itsubaki/autograd/dot"
	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

type Func func(x ...*variable.Variable) *variable.Variable

var fmap = map[string]Func{
	"sin":    F.Sin,
	"cos":    F.Cos,
	"tanh":   F.Tanh,
	"exp":    F.Exp,
	"log":    F.Log,
	"pow":    F.Pow(3.0),
	"square": F.Square,
	"neg":    F.Neg,
}

func main() {
	var order int
	var verbose bool
	var fn string
	var xval float64
	flag.IntVar(&order, "order", 1, "")
	flag.BoolVar(&verbose, "verbose", false, "")
	flag.StringVar(&fn, "func", "tanh", "")
	flag.Float64Var(&xval, "x", 1.0, "")
	flag.Parse()

	if order < 1 {
		panic("order must be greater than 0")
	}

	// input
	x := variable.New(xval)
	x.Name = "x"

	// func
	f, ok := fmap[fn]
	if !ok {
		panic(fmt.Sprintf("func %q not found", fn))
	}
	y := f(x)
	y.Name = "y"

	// backward
	y.Backward(variable.Opts{CreateGraph: true})

	for i := 1; i < order; i++ {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward(variable.Opts{CreateGraph: true})
		gx.Name = fmt.Sprintf("gx%d", i)
	}

	x.Grad.Name = fmt.Sprintf("gx%d", order)
	for _, txt := range dot.Graph(x.Grad, dot.Opt{Verbose: verbose}) {
		fmt.Println(txt)
	}
}
