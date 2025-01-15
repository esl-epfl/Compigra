//===- DiscoASM.h - Declare the functions for gen Disco ASM ------* C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares assembly generation functions for Disco-CGRA.
//
//===----------------------------------------------------------------------===//

#ifndef DISCO_CGRA_ASM_H
#define DISCO_CGRA_ASM_H

#include "compigra/ASMGen/InterferenceGraphCreation.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "compigra/Scheduler/KernelSchedule.h"
#ifdef HAVE_GUROBI
#include "gurobi_c++.h"
#endif

using namespace mlir;
using namespace compigra;

///  Allocate registers for the operations in the PEs, which are heterogeneous.
///  The register allocation is conducted under the pre-colored constraints of
///  `solution`.
LogicalResult
allocateOutRegInHeteroPE(std::map<int, mlir::Operation *> opList,
                        std::map<Operation *, ScheduleUnit> &solution,
                        unsigned maxReg,
                        std::map<int, std::unordered_set<int>> pcCtrlFlow);

#endif // DISCO_CGRA_ASM_H
