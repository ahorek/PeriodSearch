#pragma once
#include <cuda_runtime_api.h>
class Cc
{
	int deviceCcMajor;
	int deviceCcMinor;

	int GetSmxBlockCuda12() const;
	int GetSmxBlockCuda11() const;
	int GetSmxBlockCuda10() const;
	int GetSmxBlockCuda6() const;
	int GetSmxBlockCc9() const;
	int GetSmxBlockCc8() const;
	int GetSmxBlockCc7() const;
	int GetSmxBlockCc6() const;
	int GetSmxBlockCc5() const;
	int GetSmxBlockCc3() const;
	int GetSmxBlockCc2() const;
	int GetSmxBlockCc1() const;
	//void Exit() const;
	
public:

	int cudaVersion;

	explicit Cc(const cudaDeviceProp& deviceProp);

#if defined (_MSC_VER) & (_MSC_VER >= 1900) // Visual Studio 2013 or later
	Cc::~Cc() = default;
#else
	~Cc();
#endif
	
	int GetSmxBlock() const;
	void Cc::Exit() const;
};
