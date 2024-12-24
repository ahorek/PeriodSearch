#include "stdafx.h"
#if defined __GNUC__
// TODO: This has be extracted in a header file!
const int _major = 102;
const int _minor = 22;
const int _build = 1;
const int _revision = 1;
#else
#include <windows.h>
#include <tchar.h>
#endif

#if !defined __GNUC__ && defined _WIN32
/**
 * @brief Retrieves the version information of a specified file.
 *
 * This function retrieves the major, minor, build, and revision numbers from the version information
 * block of a specified file. It is designed to work on Windows systems.
 *
 * @param filename The path to the file whose version information is to be retrieved.
 * @param major A reference to an integer to store the major version number.
 * @param minor A reference to an integer to store the minor version number.
 * @param build A reference to an integer to store the build number.
 * @param revision A reference to an integer to store the revision number.
 * @return Returns true if the version information is successfully retrieved, false otherwise.
 *
 * @note This function is designed for Windows systems and uses Windows API calls.
 * @note The function buffer size is limited to 8192 bytes.
 */
bool GetVersionInfo(
    LPCTSTR filename,
    int& major,
    int& minor,
    int& build,
    int& revision)
{
    DWORD verBufferSize;
    char verBuffer[8192];

    //  Get the size of the version info block in the file
    verBufferSize = GetFileVersionInfoSize(filename, NULL);
    if (verBufferSize > 0 && verBufferSize <= sizeof(verBuffer))
    {
        //  get the version block from the file
        if (TRUE == GetFileVersionInfo(filename, NULL, verBufferSize, verBuffer))
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

#elif defined __GNUC__
/**
 * @brief Retrieves the version information for Unix-like operating systems.
 *
 * This function assigns predefined version information values to the provided references.
 * It is designed to work on Unix-like operating systems.
 *
 * @param major A reference to an integer to store the major version number.
 * @param minor A reference to an integer to store the minor version number.
 * @param build A reference to an integer to store the build number.
 * @param revision A reference to an integer to store the revision number.
 * @return Returns true as the version information is always successfully assigned.
 *
 * @note This function is designed for Unix-like operating systems and uses predefined version variables.
 */
bool GetVersionInfo(
	int& major,
	int& minor,
	int& build,
	int& revision)
{
	major = _major;
	minor = _minor;
	build = _build;
	revision = _revision;

	return true;
}

#endif
