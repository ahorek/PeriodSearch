#include <Windows.h>
#include <tchar.h>
#include <string>

bool GetVersionInfo(
    std::string filename,
    int& major,
    int& minor,
    int& build,
    int& revision)
{
    DWORD verBufferSize;
    char verBuffer[2048];

    //  Get the size of the version info block in the file
    verBufferSize = GetFileVersionInfoSize(const_cast<char*>(filename.c_str()), NULL);
    if ( verBufferSize > 0 )
    {
        //  get the version block from the file
        if (TRUE == GetFileVersionInfo(const_cast<char*>(filename.c_str()), NULL, verBufferSize, verBuffer))
        {
            UINT length;
            VS_FIXEDFILEINFO* verInfo = NULL;

            //  Query the version information for neutral language
            if (TRUE == VerQueryValue(
                verBuffer,
                _T("\\"),
                reinterpret_cast<LPVOID*>(&verInfo),
                &length))
            {
                //  Pull the version values.
                major = HIWORD(verInfo->dwProductVersionMS);
                minor = LOWORD(verInfo->dwProductVersionMS);
                build = HIWORD(verInfo->dwProductVersionLS);
                revision = LOWORD(verInfo->dwProductVersionLS);
                return true;
            }
        }
    }

    return false;
}
