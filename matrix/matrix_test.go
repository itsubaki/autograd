package matrix_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/autograd/matrix"
)

func ExampleZero() {
	for _, r := range matrix.Zero(2, 3) {
		fmt.Println(r)
	}

	// Output:
	// [0 0 0]
	// [0 0 0]
}

func ExampleZeroLike() {
	A := matrix.New(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	for _, r := range matrix.ZeroLike(A) {
		fmt.Println(r)
	}

	// Output:
	// [0 0 0]
	// [0 0 0]
}

func ExampleOneLike() {
	A := matrix.Zero(2, 3)
	for _, r := range matrix.OneLike(A) {
		fmt.Println(r)
	}

	// Output:
	// [1 1 1]
	// [1 1 1]
}

func ExampleConst() {
	fmt.Println(matrix.Const(1))

	// Output:
	// [[1]]
}

func ExampleRand() {
	shape := matrix.Shape(matrix.Rand(2, 3))
	fmt.Println(shape)

	// Output:
	// [2 3]
}

func ExampleRand_seed() {
	s := rand.NewSource(1)
	for _, r := range matrix.Rand(2, 3, s) {
		fmt.Println(r)
	}

	// Output:
	// [0.6046602879796196 0.9405090880450124 0.6645600532184904]
	// [0.4377141871869802 0.4246374970712657 0.6868230728671094]
}

func ExampleRandn() {
	shape := matrix.Shape(matrix.Randn(2, 3))
	fmt.Println(shape)

	// Output:
	// [2 3]
}

func ExampleRandn_seed() {
	s := rand.NewSource(1)
	for _, r := range matrix.Randn(2, 3, s) {
		fmt.Println(r)
	}

	// Output:
	// [-1.233758177597947 -0.12634751070237293 -0.5209945711531503]
	// [2.28571911769958 0.3228052526115799 0.5900672875996937]
}

func ExampleShape() {
	A := matrix.Zero(2, 3)
	fmt.Println(matrix.Shape(A))

	// Output:
	// [2 3]
}

func ExampleAddC() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matrix.AddC(1, A) {
		fmt.Println(r)
	}

	// Output:
	// [2 3]
	// [4 5]
}

func ExampleSubC() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matrix.SubC(1, A) {
		fmt.Println(r)
	}

	// Output:
	// [0 -1]
	// [-2 -3]
}

func ExampleMulC() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matrix.MulC(2, A) {
		fmt.Println(r)
	}

	// Output:
	// [2 4]
	// [6 8]
}

func ExampleExp() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matrix.Exp(A) {
		fmt.Println(r)
	}

	// Output:
	// [2.718281828459045 7.38905609893065]
	// [20.085536923187668 54.598150033144236]
}

func ExampleLog() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matrix.Log(A) {
		fmt.Println(r)
	}

	// Output:
	// [0 0.6931471805599453]
	// [1.0986122886681096 1.3862943611198906]
}

func ExampleSin() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matrix.Sin(A) {
		fmt.Println(r)
	}

	// Output:
	// [0.8414709848078965 0.9092974268256816]
	// [0.1411200080598672 -0.7568024953079282]
}

func ExampleCos() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matrix.Cos(A) {
		fmt.Println(r)
	}

	// Output:
	// [0.5403023058681398 -0.4161468365471424]
	// [-0.9899924966004454 -0.6536436208636119]
}

func ExampleTanh() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matrix.Tanh(A) {
		fmt.Println(r)
	}

	// Output:
	// [0.7615941559557649 0.9640275800758169]
	// [0.9950547536867305 0.999329299739067]
}

func ExamplePow() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matrix.Pow(2, A) {
		fmt.Println(r)
	}

	// Output:
	// [1 4]
	// [9 16]
}

func ExampleAdd() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range matrix.Add(A, B) {
		fmt.Println(r)
	}

	// Output:
	// [6 8]
	// [10 12]
}

func ExampleSub() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{10, 20},
		[]float64{30, 40},
	)

	for _, r := range matrix.Sub(A, B) {
		fmt.Println(r)
	}

	// Output:
	// [-9 -18]
	// [-27 -36]
}

func ExampleMul() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)
	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range matrix.Mul(A, B) {
		fmt.Println(r)
	}

	// Output:
	// [5 12]
	// [21 32]
}

func ExampleDiv() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)
	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range matrix.Div(A, B) {
		fmt.Println(r)
	}

	// Output:
	// [0.2 0.3333333333333333]
	// [0.42857142857142855 0.5]
}

func ExampleDot() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range matrix.Dot(A, B) {
		fmt.Println(r)
	}

	// Output:
	// [19 22]
	// [43 50]
}

func ExampleMaxAxis1() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
		[]float64{5, 2},
	)

	for _, r := range matrix.MaxAxis1(A) {
		fmt.Println(r)
	}

	// Output:
	// [2]
	// [4]
	// [5]
}

func ExampleMax() {
	A := matrix.New(
		[]float64{1, 5},
		[]float64{3, 4},
	)

	fmt.Println(matrix.Max(A))

	// Output:
	// 5
}

func ExampleMin() {
	A := matrix.New(
		[]float64{1, -5},
		[]float64{3, 4},
	)

	fmt.Println(matrix.Min(A))

	// Output:
	// -5
}

func ExampleClip() {
	A := matrix.New(
		[]float64{-3, -2, -1, 0, 1, 2, 3, 4, 5, 6},
		[]float64{7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
	)

	for _, r := range matrix.Clip(A, 0, 10) {
		fmt.Println(r)
	}

	// Output:
	// [0 0 0 0 1 2 3 4 5 6]
	// [7 8 9 10 10 10 10 10 10 10]
}

func ExampleMask() {
	A := matrix.New(
		[]float64{-1, 2},
		[]float64{3, -4},
	)

	for _, r := range matrix.Mask(A, func(v float64) bool { return v > 0 }) {
		fmt.Println(r)
	}

	// Output:
	// [0 1]
	// [1 0]
}

func ExampleBroadcastTo() {
	A := matrix.New([]float64{1})

	for _, r := range matrix.BroadcastTo([]int{3, 5}, A) {
		fmt.Println(r)
	}

	// Output:
	// [1 1 1 1 1]
	// [1 1 1 1 1]
	// [1 1 1 1 1]
}

func ExampleBroadcastTo_row() {
	A := matrix.New([]float64{1, 2})

	for _, r := range matrix.BroadcastTo([]int{5, -1}, A) {
		fmt.Println(r)
	}

	// Output:
	// [1 2]
	// [1 2]
	// [1 2]
	// [1 2]
	// [1 2]
}

func ExampleBroadcastTo_column() {
	A := matrix.New(
		[]float64{1},
		[]float64{2},
	)

	for _, r := range matrix.BroadcastTo([]int{-1, 5}, A) {
		fmt.Println(r)
	}

	// Output:
	// [1 1 1 1 1]
	// [2 2 2 2 2]
}

func ExampleBroadcastTo_noEffect() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matrix.BroadcastTo([]int{2, 2}, A) {
		fmt.Println(r)
	}

	// Output:
	// [1 2]
	// [3 4]
}

func ExampleBroadcast() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)
	B := matrix.New(
		[]float64{1, 2},
	)

	AA, BB := matrix.Broadcast(A, B)
	for _, r := range AA {
		fmt.Println(r)
	}
	for _, r := range BB {
		fmt.Println(r)
	}

	// Output:
	// [1 2]
	// [3 4]
	// [1 2]
	// [1 2]
}

func ExampleBroadcast_constA() {
	A := matrix.Const(1)
	B := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	AA, BB := matrix.Broadcast(A, B)
	for _, r := range AA {
		fmt.Println(r)
	}
	for _, r := range BB {
		fmt.Println(r)
	}

	// Output:
	// [1 1]
	// [1 1]
	// [1 2]
	// [3 4]
}

func ExampleBroadcast_constB() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)
	B := matrix.Const(1)

	AA, BB := matrix.Broadcast(A, B)
	for _, r := range AA {
		fmt.Println(r)
	}
	for _, r := range BB {
		fmt.Println(r)
	}

	// Output:
	// [1 2]
	// [3 4]
	// [1 1]
	// [1 1]
}

func ExampleSum() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(matrix.Sum(A))

	// Output:
	// 10
}

func ExampleSumTo_sum() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(matrix.SumTo([]int{1, 1}, A))

	// Output:
	// [[10]]
}

func ExampleSumTo_axis0() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(matrix.SumTo([]int{1, 2}, A))

	// Output:
	// [[4 6]]
}

func ExampleSumTo_axis1() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(matrix.SumTo([]int{2, 1}, A))

	// Output:
	// [[3] [7]]
}

func ExampleSumTo_noeffect() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(matrix.SumTo([]int{2, 3}, A))

	// Output:
	// [[1 2] [3 4]]
}

func ExampleSumAxis0() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matrix.SumAxis0(A) {
		fmt.Println(r)
	}

	// Output:
	// [4 6]
}

func ExampleSumAxis1() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matrix.SumAxis1(A) {
		fmt.Println(r)
	}

	// Output:
	// [3]
	// [7]
}

func ExampleReshape() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(matrix.Reshape([]int{1, 4}, A))
	fmt.Println(matrix.Reshape([]int{4, 1}, A))
	fmt.Println(matrix.Reshape([]int{2, 2}, A))
	fmt.Println()

	fmt.Println(matrix.Reshape([]int{1, -1}, A))
	fmt.Println(matrix.Reshape([]int{4, -1}, A))
	fmt.Println(matrix.Reshape([]int{2, -1}, A))
	fmt.Println()

	fmt.Println(matrix.Reshape([]int{-1, 1}, A))
	fmt.Println(matrix.Reshape([]int{-1, 4}, A))
	fmt.Println(matrix.Reshape([]int{-1, 2}, A))
	fmt.Println()

	// Output:
	// [[1 2 3 4]]
	// [[1] [2] [3] [4]]
	// [[1 2] [3 4]]
	//
	// [[1 2 3 4]]
	// [[1] [2] [3] [4]]
	// [[1 2] [3 4]]
	//
	// [[1] [2] [3] [4]]
	// [[1 2 3 4]]
	// [[1 2] [3 4]]
}
