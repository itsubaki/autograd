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
	"cos":    F.Cos,
	"exp":    F.Exp,
	"neg":    F.Neg,
	"pow":    F.Pow(3.0),
	"sin":    F.Sin,
	"square": F.Square,
	"tanh":   F.Tanh,
}

func main() {
	var order int
	var verbose bool
	var fn string
	flag.IntVar(&order, "order", 1, "")
	flag.BoolVar(&verbose, "verbose", false, "")
	flag.StringVar(&fn, "func", "tanh", "")
	flag.Parse()

	if order < 1 {
		panic("order must be greater than 0")
	}

	// input
	x := variable.New(1.0)
	x.Name = "x"

	// func
	f, ok := fmap[fn]
	if !ok {
		panic(fmt.Sprintf("func %q not found", fn))
	}
	y := f(x)
	y.Name = "y"

	// backward
	y.Backward()
	y.Grad.Name = "gy"

	for i := 0; i < order-1; i++ {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward()
	}

	x.Grad.Name = fmt.Sprintf("gx%d", order)
	for _, txt := range dot.Graph(x.Grad, dot.Opt{Verbose: verbose}) {
		fmt.Println(txt)
	}
}
