$source = @"
using System;
using System.IO;
using System.Collections.Generic;

public class TiffLzwDecoder
{
    public static byte[] DecodeTile(byte[] src, int srcOffset, int srcLength, int maxDstLength)
    {
        byte[] dst = new byte[maxDstLength];
        int dstPos = 0;

        byte[][] stringTable = new byte[4096][];
        int tableEntries = 258;
        for (int i = 0; i < 256; i++)
        {
            stringTable[i] = new byte[] { (byte)i };
        }

        int bitPos = 0;
        int codeSize = 9;

        Func<int> readCode = () =>
        {
            int byteIndex = srcOffset + (bitPos >> 3);
            int bitOffset = bitPos & 7;
            if (byteIndex + 3 >= src.Length) return 257; // EOI

            uint b0 = src[byteIndex];
            uint b1 = src[byteIndex + 1];
            uint b2 = src[byteIndex + 2];
            uint b3 = (byteIndex + 3 < src.Length) ? src[byteIndex + 3] : (uint)0;

            uint buffer = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
            buffer <<= bitOffset;
            uint code = buffer >> (32 - codeSize);

            bitPos += codeSize;
            return (int)code;
        };

        Action resetTable = () =>
        {
            tableEntries = 258;
            codeSize = 9;
        };

        byte[] oldEntry = null;

        while (dstPos < maxDstLength)
        {
            int code = readCode();
            if (code == 257) break; // EOI
            if (code == 256) // Clear
            {
                resetTable();
                code = readCode();
                if (code == 257) break;
                byte[] entry = stringTable[code];
                Array.Copy(entry, 0, dst, dstPos, entry.Length);
                dstPos += entry.Length;
                oldEntry = entry;
                continue;
            }

            byte[] currentEntry;
            if (code < tableEntries)
            {
                currentEntry = stringTable[code];
            }
            else if (code == tableEntries && oldEntry != null)
            {
                currentEntry = new byte[oldEntry.Length + 1];
                Array.Copy(oldEntry, 0, currentEntry, 0, oldEntry.Length);
                currentEntry[oldEntry.Length] = oldEntry[0];
            }
            else
            {
                break;
            }

            int toCopy = Math.Min(currentEntry.Length, maxDstLength - dstPos);
            Array.Copy(currentEntry, 0, dst, dstPos, toCopy);
            dstPos += toCopy;

            if (oldEntry != null && tableEntries < 4096)
            {
                byte[] newEntry = new byte[oldEntry.Length + 1];
                Array.Copy(oldEntry, 0, newEntry, 0, oldEntry.Length);
                newEntry[oldEntry.Length] = currentEntry[0];
                stringTable[tableEntries++] = newEntry;

                if (tableEntries == 511) codeSize = 10;
                else if (tableEntries == 1023) codeSize = 11;
                else if (tableEntries == 2047) codeSize = 12;
            }

            oldEntry = currentEntry;
        }

        return dst;
    }
}
"@

Add-Type -TypeDefinition $source -Language CSharp

$filePath = "temp_audit_samples\images\00000.tif"
$bytes = [System.IO.File]::ReadAllBytes($filePath)

# Tile 0 offset and length: 5659644, 2238744
$t0_off = 5659644
$t0_len = 2238744
# Expected uncompressed tile: 512 x 512 x 2 bands x 4 bytes = 2,097,152 bytes
$expectedBytes = 512 * 512 * 2 * 4

Write-Output "Decompressing Tile 0..."
$decompressed = [TiffLzwDecoder]::DecodeTile($bytes, $t0_off, $t0_len, $expectedBytes)
Write-Output "Decompressed bytes: $($decompressed.Length)"

# Sample float32 values:
# TIFF interleaved: Pixel 0: Band0 (float32), Band1 (float32)
$band0_vals = @()
$band1_vals = @()

$min0 = [float]::MaxValue; $max0 = [float]::MinValue
$min1 = [float]::MaxValue; $max1 = [float]::MinValue
$nanCount = 0; $infCount = 0

for ($p = 0; $p -lt (512*512); $p++) {
    $idx = $p * 8
    # BigEndian float32
    $b0 = @($decompressed[$idx+3], $decompressed[$idx+2], $decompressed[$idx+1], $decompressed[$idx])
    $f0 = [System.BitConverter]::ToSingle($b0, 0)
    
    $b1 = @($decompressed[$idx+7], $decompressed[$idx+6], $decompressed[$idx+5], $decompressed[$idx+4])
    $f1 = [System.BitConverter]::ToSingle($b1, 0)
    
    if ([float]::IsNaN($f0) -or [float]::IsNaN($f1)) { $nanCount++ }
    if ([float]::IsInfinity($f0) -or [float]::IsInfinity($f1)) { $infCount++ }
    
    if ($f0 -lt $min0) { $min0 = $f0 }
    if ($f0 -gt $max0) { $max0 = $f0 }
    if ($f1 -lt $min1) { $min1 = $f1 }
    if ($f1 -gt $max1) { $max1 = $f1 }
    
    if ($p -lt 5) {
        $band0_vals += $f0
        $band1_vals += $f1
    }
}

Write-Output "=== TILE 0 SAR BACKSCATTER PIXEL STATS ==="
Write-Output "First 5 Band 0 (VH dB): $($band0_vals -join ', ')"
Write-Output "First 5 Band 1 (VV dB): $($band1_vals -join ', ')"
Write-Output "Band 0 (VH dB) Range: Min = $min0 dB, Max = $max0 dB"
Write-Output "Band 1 (VV dB) Range: Min = $min1 dB, Max = $max1 dB"
Write-Output "NaN Count in Tile 0: $nanCount"
Write-Output "Inf Count in Tile 0: $infCount"
