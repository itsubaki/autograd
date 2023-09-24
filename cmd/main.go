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
	flag.IntVar(&iter, "iter", 0, "")
	flag.Parse()

	// input
	x := variable.New(1.0)
	x.Name = "x"

	// func
	y := F.Tanh(x)
	y.Name = "y"
	y.Backward()
	y.Grad.Name = "gy"

	for i := 0; i < iter; i++ {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward()
	}

	gx := x.Grad
	gx.Name = fmt.Sprintf("gx%d", iter)
	for _, txt := range dot.Graph(gx) {
		fmt.Println(txt)
	}
}
