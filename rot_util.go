package main

import (
	"fmt"
	"runtime"
)

func lRot(a []float64, rotation int) []float64 {
	size := len(a)
	var newArray []float64
	for i := 0; i < rotation; i++ {
		newArray = a[1:size]
		newArray = append(newArray, a[0])
		a = newArray
	}
	return a
}

func rRot(a []float64, rotation int) []float64 {
	return lRot(a, len(a)-rotation)
}

func addSlice(a []float64, b []float64) []float64 {
	c := make([]float64, len(a))
	for i := range a {
		c[i] = a[i] + b[i]
	}
	return c
}

func keep_vec(input []float64, in_wid, kp_wid, ul int) []float64 {
	output := make([]float64, len(input))

	tmp := gen_keep_vec(len(input), in_wid, kp_wid, ul)

	for i := range output {
		output[i] = input[i] * float64(tmp[i])
	}

	return output
}

func keep_vec_stride(input []float64, in_wid, kp_wid, step, ul int, raw_in_wid_odd bool) []float64 {
	output := make([]float64, len(input))

	tmp := gen_keep_vec_stride(len(input), in_wid, kp_wid, step, ul, raw_in_wid_odd)

	for i := range output {
		output[i] = input[i] * float64(tmp[i])
	}

	return output
}

func gen_keep_vec(vec_size, in_wid, kp_wid, ul int) (idx []int) {
	logN := 0
	for ; (1 << logN) < (2 * vec_size); logN++ {
	}
	idx = make([]int, vec_size)
	batch := 2 * vec_size / (in_wid * in_wid)
	if kp_wid < in_wid/2 {
		panic("keep width too small. less than in_wid/2")
	}

	if ul == 0 {
		for i := 0; i < in_wid/2; i++ {
			for j := 0; j < kp_wid; j++ {
				for b := 0; b < batch; b++ {
					id := int(reverseBits(uint32(in_wid*batch*i+batch*j+b), logN-1))
					idx[id] = 1
				}
			}
		}
	} else if ul == 1 {
		for i := 0; i < kp_wid-in_wid/2; i++ {
			for j := 0; j < kp_wid; j++ {
				for b := 0; b < batch; b++ {
					id := int(reverseBits(uint32(in_wid*batch*i+batch*j+b), logN-1))
					idx[id] = 1
				}
			}
		}
	} else {
		panic("ul not 0 nor 1")
	}

	return idx
}

func gen_keep_vec_sparse(vec_size, in_wid, kp_wid, log_sparse int) (idx []int) {
	logN := 0
	for ; (1 << logN) < (2 * vec_size); logN++ {
	}
	idx = make([]int, vec_size)
	batch := 2 * vec_size / (in_wid * in_wid)
	sparsity := 1 << log_sparse
	if sparsity == 1 {
		panic("We do not support full packing in gen_keep_vec_sparse")
	}
	if kp_wid < in_wid/2 {
		panic("keep width too small. less than in_wid/2")
	}

	for i := 0; i < in_wid/2; i++ {
		for j := 0; j < kp_wid; j++ {
			for b := 0; b < batch/sparsity; b++ {
				id := int(reverseBits(uint32(in_wid*batch*i+batch*j+b*sparsity), logN-1))
				idx[id] = 1
			}
		}
	}
	for i := 0; i < kp_wid-in_wid/2; i++ {
		for j := 0; j < kp_wid; j++ {
			for b := 0; b < batch/sparsity; b++ {
				id := int(reverseBits(uint32(in_wid*batch*i+batch*j+b*sparsity), logN-1)) + vec_size/sparsity
				idx[id] = 1
			}
		}
	}

	post_slot := 2 * len(idx) / sparsity
	for i := 0; i < post_slot; i++ {
		for j := 1; j < sparsity/2; j++ {
			idx[i+post_slot*j] = idx[i]
		}
	}

	return idx
}

func gen_keep_vec_stride(vec_size, in_wid, kp_wid, step, ul int, raw_in_wid_odd bool) (idx []int) {
	logN := 0
	for ; (1 << logN) < (2 * vec_size); logN++ {
	}
	idx = make([]int, vec_size)
	batch := 2 * vec_size / (in_wid * in_wid)

	var init int
	if raw_in_wid_odd {
		init = 0
	} else {
		init = step - 1
	}

	if ul == 0 {
		for i := 0; i < kp_wid; i++ {
			if (init + i*step) < in_wid/2 {
				for j := 0; j < kp_wid; j++ {
					for b := 0; b < batch; b++ {
						id := int(reverseBits(uint32(in_wid*batch*(init+i*step)+batch*(j*step+init)+b), logN-1))
						idx[id] = 1
					}
				}
			}
		}
	} else if ul == 1 {
		for i := 0; i < kp_wid; i++ {
			if (init + i*step) >= in_wid/2 {
				for j := 0; j < kp_wid; j++ {
					for b := 0; b < batch; b++ {
						id := int(reverseBits(uint32(in_wid*batch*(init+i*step-in_wid/2)+batch*(j*step+init)+b), logN-1))
						idx[id] = 1
					}
				}
			}
		}
	} else {
		panic("ul not 0 nor 1")
	}

	return idx
}

func gen_comprs_fast(vec_size, in_wid, kp_wid, pos, ul int) (m_idx, r_idx map[int][]int) {
	m_idx = make(map[int][]int)
	r_idx = make(map[int][]int)
	batch := 2 * vec_size / (in_wid * in_wid)

	if kp_wid < in_wid/2 {
		panic("keep width too small. less than in_wid/2")
	}
	pos = int(reverseBits(uint32(pos), 2))
	min_wid := in_wid / 4
	if in_wid%4 != 0 {
		panic("input wid not divisible by 4")
	}
	if in_wid%2 != 0 {
		panic("input wid not divisible by 2")
	}
	log_in_wid := 0
	for ; (1 << log_in_wid) < in_wid; log_in_wid++ {
	}

	for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
		tmp := make([]int, vec_size)
		for b := 0; b < batch; b++ {
			for i := 0; i < min_wid; i++ {
				if (ul == 0) && (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) {
					idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
					tmp[idx] = 1
				}
				if (ul == 1) && (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) && (reverseBits(uint32(min_wid+i), log_in_wid-1) < uint32(kp_wid-in_wid/2)) {
					idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
					tmp[idx] = 1
				}
			}
		}
		rot := j*min_wid - 2*min_wid*min_wid + min_wid
		m_idx[rot] = tmp
	}
	for b := 0; b < batch; b++ { // kinds of mov depends on b
		tmp := make([]int, vec_size)
		for j := 0; j < 2*min_wid; j++ {
			for i := 0; i < min_wid; i++ {
				idx := 2*min_wid*in_wid*b + 3*in_wid/2*min_wid + j*min_wid + i
				tmp[idx] = 1
			}
		}
		rot := 3*b*min_wid*in_wid/2 - pos*min_wid*in_wid/2*batch + 3*min_wid*in_wid/2
		r_idx[rot] = tmp
	}

	return m_idx, r_idx
}

func gen_comprs_sparse(vec_size, in_wid, kp_wid, log_sparse, ul, pos int) (m_idx, r_idx map[int][]int) {
	m_idx = make(map[int][]int)
	r_idx = make(map[int][]int)
	batch := 2 * vec_size / (in_wid * in_wid * (1 << log_sparse))

	min_wid := in_wid / 2
	if in_wid%2 != 0 {
		panic("input wid not divisible by 2")
	}
	log_in_wid := 0
	for ; (1 << log_in_wid) < in_wid; log_in_wid++ {
	}

	if log_sparse != 0 {
		if pos != 0 {
			panic("No pos != 0 cases for log_sparse != 0")
		}
		for j := 0; j < min_wid; j++ {
			tmp := make([]int, vec_size)
			for b := 0; b < batch; b++ {
				for i := 0; i < min_wid/2; i++ {
					for k := 0; k < 2; k++ {
						if (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && ((reverseBits(uint32(i), log_in_wid-2) + uint32(k)*uint32(min_wid)/2) < uint32(kp_wid)) {
							idx := k*in_wid*min_wid*batch + in_wid*in_wid*b/2 + in_wid*j/2 + i
							tmp[idx] = 1
						}
					}
				}
			}
			for i := 0; i < vec_size/(1<<(log_sparse-1)); i++ {
				for k := 1; k < (1 << (log_sparse - 1)); k++ {
					tmp[i+k*vec_size/(1<<(log_sparse-1))] = tmp[i]
				}
			}
			rot := j * min_wid / 2
			m_idx[rot] = tmp
		}

		for b := 0; b < batch; b++ {
			tmp := make([]int, vec_size)
			for j := 0; j < min_wid; j++ {
				for i := 0; i < min_wid/2; i++ {
					for k := 0; k < 2; k++ {
						idx := k*in_wid*min_wid*batch + b*in_wid*in_wid/2 + j*min_wid/2 + i
						tmp[idx] = 1
					}
				}
			}
			for i := 0; i < vec_size/(1<<(log_sparse-1)); i++ {
				for k := 1; k < (1 << (log_sparse - 1)); k++ {
					tmp[i+k*vec_size/(1<<(log_sparse-1))] = tmp[i]
				}
			}
			rot := 3 * b * min_wid * min_wid / 2
			r_idx[rot] = tmp
		}
	} else {
		if batch > 8*min_wid {
			for j := 0; j < min_wid; j++ {
				for bk := 0; bk < 8; bk++ {
					tmp := make([]int, vec_size)
					for b := 0; b < batch/8; b++ {
						for i := 0; i < min_wid/2; i++ {
							if (ul == 0) && (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-2) < uint32(kp_wid)) {
								idx := 8*in_wid*min_wid*b + bk*min_wid*in_wid + min_wid*j + i
								tmp[idx] = 1
							}
							if (ul == 1) && (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-2)+uint32(min_wid/2) < uint32(kp_wid)) {
								idx := 8*in_wid*min_wid*b + bk*min_wid*in_wid + min_wid*j + i
								tmp[idx] = 1
							}
						}
					}
					rot := j*min_wid/2 + 7*bk*min_wid*min_wid/2
					m_idx[rot] = tmp
				}
			}

			for b := 0; b < batch/8; b++ {
				tmp := make([]int, vec_size)
				for bk := 0; bk < 8; bk++ {
					for j := 0; j < min_wid; j++ {
						for i := 0; i < min_wid/2; i++ {
							idx := 8*b*in_wid*min_wid + bk*min_wid*min_wid/2 + j*min_wid/2 + i
							tmp[idx] = 1
						}
					}
				}
				rot := 3*b*8*min_wid*min_wid/2 - int(reverseBits(uint32(pos), 2))*batch*min_wid*min_wid/2
				r_idx[rot] = tmp
			}
		} else if batch > 4*min_wid {
			for j := 0; j < min_wid; j++ {
				for bk := 0; bk < 4; bk++ {
					tmp := make([]int, vec_size)
					for b := 0; b < batch/4; b++ {
						for i := 0; i < min_wid/2; i++ {
							if (ul == 0) && (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-2) < uint32(kp_wid)) {
								idx := 4*in_wid*min_wid*b + bk*min_wid*in_wid + min_wid*j + i
								tmp[idx] = 1
							}
							if (ul == 1) && (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-2)+uint32(min_wid/2) < uint32(kp_wid)) {
								idx := 4*in_wid*min_wid*b + bk*min_wid*in_wid + min_wid*j + i
								tmp[idx] = 1
							}
						}
					}
					rot := j*min_wid/2 + 3*bk*min_wid*min_wid/2
					m_idx[rot] = tmp
				}
			}

			for b := 0; b < batch/4; b++ {
				tmp := make([]int, vec_size)
				for bk := 0; bk < 4; bk++ {
					for j := 0; j < min_wid; j++ {
						for i := 0; i < min_wid/2; i++ {
							idx := 4*b*in_wid*min_wid + bk*min_wid*min_wid/2 + j*min_wid/2 + i
							tmp[idx] = 1
						}
					}
				}
				rot := 3*b*4*min_wid*min_wid/2 - int(reverseBits(uint32(pos), 2))*batch*min_wid*min_wid/2
				r_idx[rot] = tmp
			}
		} else {
			for j := 0; j < min_wid; j++ {
				tmp := make([]int, vec_size)
				for b := 0; b < batch; b++ {
					for i := 0; i < min_wid/2; i++ {
						if (ul == 0) && (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-2) < uint32(kp_wid)) {
							idx := in_wid*min_wid*b + min_wid*j + i
							tmp[idx] = 1
						}
						if (ul == 1) && (reverseBits(uint32(j), log_in_wid-1) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-2)+uint32(min_wid/2) < uint32(kp_wid)) {
							idx := in_wid*min_wid*b + min_wid*j + i
							tmp[idx] = 1
						}
					}
				}
				rot := j * min_wid / 2
				m_idx[rot] = tmp
			}

			for b := 0; b < batch; b++ {
				tmp := make([]int, vec_size)
				for j := 0; j < min_wid; j++ {
					for i := 0; i < min_wid/2; i++ {
						idx := b*in_wid*min_wid + j*min_wid/2 + i
						tmp[idx] = 1
					}
				}
				rot := 3*b*min_wid*min_wid/2 - int(reverseBits(uint32(pos), 2))*batch*min_wid*min_wid/2
				r_idx[rot] = tmp
			}
		}
	}

	return m_idx, r_idx
}

// memory tracking
func printMemUsage(s string) float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("%s Memory Allocation = %.2f GB \n", s, bToGb(m.Alloc))
	if m.Alloc > allocMem {
		allocMem = m.Alloc
	}

	return bToGb(m.Alloc)
}

func bToGb(b uint64) float64 {
	return float64(b) / 1024 / 1024 / 1024
}
