//go:build !(darwin && arm64)

package tensor

func matmul(a, b, o []float64, m, k, n int) {
	for i := range m {
		ai := i * k
		oi := i * n

		for p := range k {
			aip := a[ai+p]
			bp := p * n

			for j := range n {
				o[oi+j] += aip * b[bp+j]
			}
		}
	}
}
