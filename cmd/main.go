package main

import (
	"flag"
	"fmt"

	"github.com/itsubaki/autograd/dot"
	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func main() {
	var iter int
	var verbose bool
	flag.IntVar(&iter, "iter", 0, "")
	flag.BoolVar(&verbose, "verbose", false, "")
	flag.Parse()

	// input
	x := variable.New(1.0)
	x.Name = "x"

	// func
	y := F.Tanh(x)
	y.Name = "y"

	// backward
	y.Backward()
	y.Grad.Name = "gy"

	for i := 0; i < iter; i++ {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward()
	}

	x.Grad.Name = fmt.Sprintf("gx%d", iter+1)
	for _, txt := range dot.Graph(x.Grad, dot.Opt{Verbose: verbose}) {
		fmt.Println(txt)
	}
}
