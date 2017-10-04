#ifdef RW_HEADER
#include "llvm/Bitcode/ReaderWriter.h"
#else
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#endif
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"

#include <system_error>

using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::value_desc("filename"));

int main(int argc, char **argv) {
  LLVMContext Context;
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "amdgcn device RTL: set linkage tool\n");

  std::string ErrorMessage;
  Module *M = nullptr;

  {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(InputFilename);
    std::unique_ptr<MemoryBuffer> &BufferPtr = BufferOrErr.get();
    if (std::error_code  ec = BufferOrErr.getError())
      ErrorMessage = ec.message();
    else {
#ifdef RW_HEADER
      ErrorOr<std::unique_ptr<Module>> ModuleOrErr =
          parseBitcodeFile(BufferPtr.get()->getMemBufferRef(), Context);
#else
      ErrorOr<std::unique_ptr<Module>> ModuleOrErr =
          expectedToErrorOrAndEmitErrors(Context,
          parseBitcodeFile(BufferPtr.get()->getMemBufferRef(), Context));
#endif
      if (std::error_code ec = ModuleOrErr.getError())
        ErrorMessage = ec.message();

      M = ModuleOrErr.get().release();
    }
  }

  if (!M) {
    errs() << argv[0] << ": ";
    if (ErrorMessage.size())
      errs() << ErrorMessage << "\n";
    else
      errs() << "bitcode didn't read correctly.\n";
    return 1;
  }

  // Set alwaysinline for every external functions
  for (auto curFref = M->getFunctionList().begin(), endFref = M->getFunctionList().end();
      curFref != endFref; ++curFref) {
    if (! curFref->isDeclaration()) {
      if (curFref->hasFnAttribute(llvm::Attribute::NoInline))
        curFref->removeFnAttr(llvm::Attribute::NoInline);
      curFref->addFnAttr(llvm::Attribute::AlwaysInline);
    }
  }

  // Set linkage of every external definition to linkonce_odr.
  for (Module::iterator i = M->begin(), e = M->end(); i != e; ++i) {
    if (!i->isDeclaration() && i->getLinkage() == GlobalValue::ExternalLinkage)
      i->setLinkage(GlobalValue::LinkOnceODRLinkage);
  }

  for (Module::global_iterator i = M->global_begin(), e = M->global_end();
       i != e; ++i) {
    if (!i->isDeclaration() && i->getLinkage() == GlobalValue::ExternalLinkage)
      i->setLinkage(GlobalValue::LinkOnceODRLinkage);
    if (i->getType()->getPointerAddressSpace() == 3) {
      if (i->hasInitializer())
        i->setInitializer(llvm::UndefValue::get(i->getType()->getElementType()));
      if (i->getLinkage() != GlobalValue::WeakAnyLinkage)
        i->setLinkage(GlobalValue::WeakAnyLinkage);
    }
  }

  if (OutputFilename.empty()) {
    errs() << "no output file\n";
    return 1;
  }

  std::error_code EC;
  std::unique_ptr<tool_output_file> Out
  (new tool_output_file(OutputFilename, EC, sys::fs::F_None));
  if (EC) {
    errs() << EC.message() << '\n';
    exit(1);
  }

  WriteBitcodeToFile(M, Out->os());

  // Declare success.
  Out->keep();
  return 0;
}

