#property copyright "Cascade Research"
#property link      "https://cascade.research"
#property version   "1.00"
#property strict

#ifndef ERR_FILE_ALREADY_EXISTS
#define ERR_FILE_ALREADY_EXISTS 5004
#endif

// Competition FX pairs - GBP/JPY primary for max volatility
input string         InpOutputFile   = "results\\live_fx.csv";
input ENUM_TIMEFRAMES InpTF          = PERIOD_M5;
input int            InpTimerSeconds = 10;
input string         InpSymbols      = "GBPJPY,USDJPY,GBPUSD";  // High-volatility FX pairs

const string CSV_HEADER = "symbol,timestamp,open,high,low,close,volume";

string g_symbols[];
int    g_symbol_count = 0;
datetime g_last_exported[];  // Per-symbol last exported timestamp

int OnInit()
{
   // Parse symbol list
   g_symbol_count = StringSplit(InpSymbols, ',', g_symbols);
   if(g_symbol_count <= 0)
   {
      PrintFormat("No symbols configured. Set InpSymbols input.");
      return INIT_FAILED;
   }
   
   ArrayResize(g_last_exported, g_symbol_count);
   ArrayInitialize(g_last_exported, 0);
   
   if(!EnsureOutputDirectories())
      return INIT_FAILED;

   if(!LoadLastExportedTimestamps())
      return INIT_FAILED;

   int timer_period = MathMax(InpTimerSeconds, 1);
   EventSetTimer(timer_period);

   PrintFormat("FXSymbolExporter initialized | file=%s tf=%d symbols=%s", 
               InpOutputFile, InpTF, InpSymbols);
   
   for(int i = 0; i < g_symbol_count; i++)
   {
      string last_label = g_last_exported[i] > 0
         ? TimeToString(g_last_exported[i], TIME_DATE | TIME_MINUTES)
         : "none";
      PrintFormat("  Symbol[%d]: %s last_bar=%s", i, g_symbols[i], last_label);
   }

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTick()
{
   // No strategy logic; export-only agent.
}

void OnTimer()
{
   for(int i = 0; i < g_symbol_count; i++)
   {
      string symbol = g_symbols[i];
      StringTrimLeft(symbol);
      StringTrimRight(symbol);
      
      ExportSymbolBar(symbol, i);
   }
}

void ExportSymbolBar(const string symbol, const int idx)
{
   const int bars = iBars(symbol, InpTF);
   if(bars < 2)
      return;

   datetime bar_time = iTime(symbol, InpTF, 1); // last CLOSED bar
   if(bar_time <= g_last_exported[idx])
      return;

   double open  = iOpen(symbol, InpTF, 1);
   double high  = iHigh(symbol, InpTF, 1);
   double low   = iLow(symbol, InpTF, 1);
   double close = iClose(symbol, InpTF, 1);
   long volume  = (long)iVolume(symbol, InpTF, 1);

   string timestamp = TimeToString(bar_time, TIME_DATE | TIME_MINUTES);
   string line = StringFormat(
      "%s,%s,%.6f,%.6f,%.6f,%.6f,%I64d\n",
      symbol,
      timestamp,
      open,
      high,
      low,
      close,
      volume
   );

   if(AppendLineAtomically(InpOutputFile, line))
   {
      g_last_exported[idx] = bar_time;
      PrintFormat("live_bar_exported | symbol=%s time=%s", symbol, timestamp);
   }
}

bool EnsureOutputDirectories()
{
   string normalized = InpOutputFile;
   StringReplace(normalized, "/", "\\");

   int last_sep = -1;
   const int len = StringLen(normalized);
   for(int i = 0; i < len; i++)
   {
      ushort ch = StringGetCharacter(normalized, i);
      if(ch == '\\')
         last_sep = i;
   }

   if(last_sep <= 0)
      return true;

   string folder = StringSubstr(normalized, 0, last_sep);
   string parts[];
   int count = StringSplit(folder, '\\', parts);
   if(count <= 0)
      return true;

   string partial = "";
   for(int i = 0; i < count; i++)
   {
      if(StringLen(parts[i]) == 0)
         continue;

      if(StringLen(partial) == 0)
         partial = parts[i];
      else
         partial = partial + "\\" + parts[i];

      if(!EnsureFolderExists(partial))
         return false;
   }

   return true;
}

bool EnsureFolderExists(const string path)
{
   ResetLastError();
   if(FolderCreate(path))
      return true;

   int err = GetLastError();
   if(err == ERR_FILE_ALREADY_EXISTS)
      return true;

   PrintFormat("folder_create_failed | path=%s code=%d", path, err);
   return false;
}

bool LoadLastExportedTimestamps()
{
   if(!FileIsExist(InpOutputFile))
      return true;

   int handle = FileOpen(InpOutputFile, FILE_READ | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      PrintFormat("last_timestamp_load_failed | file=%s code=%d", InpOutputFile, GetLastError());
      return false;
   }

   // Read all lines and track last timestamp per symbol
   while(!FileIsEnding(handle))
   {
      string line = FileReadString(handle);
      if(StringLen(line) == 0)
         continue;
      if(StringSubstr(line, 0, 6) == "symbol")
         continue;  // Skip header

      string fields[];
      const int count = StringSplit(line, ',', fields);
      if(count < 2)
         continue;

      string sym = fields[0];
      datetime parsed = StringToTime(fields[1]);
      
      // Find symbol index and update last timestamp
      for(int i = 0; i < g_symbol_count; i++)
      {
         string target = g_symbols[i];
         StringTrimLeft(target);
         StringTrimRight(target);
         if(sym == target && parsed > g_last_exported[i])
            g_last_exported[i] = parsed;
      }
   }

   FileClose(handle);
   return true;
}

bool AppendLineAtomically(const string final_path, const string line)
{
   const string tmp_path = final_path + ".tmp";

   if(FileIsExist(tmp_path))
      FileDelete(tmp_path);

   if(FileIsExist(final_path))
   {
      if(!FileCopy(final_path, 0, tmp_path, 0))
      {
         PrintFormat("file_copy_failed | src=%s dst=%s code=%d", final_path, tmp_path, GetLastError());
         return false;
      }
   }
   else
   {
      int tmp_handle = FileOpen(tmp_path, FILE_WRITE | FILE_TXT | FILE_ANSI);
      if(tmp_handle == INVALID_HANDLE)
      {
         PrintFormat("tmp_file_create_failed | file=%s code=%d", tmp_path, GetLastError());
         return false;
      }
      FileWriteString(tmp_handle, CSV_HEADER + "\n");
      FileClose(tmp_handle);
   }

   int append_handle = FileOpen(tmp_path, FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(append_handle == INVALID_HANDLE)
   {
      PrintFormat("tmp_file_open_failed | file=%s code=%d", tmp_path, GetLastError());
      return false;
   }

   FileSeek(append_handle, 0, SEEK_END);
   FileWriteString(append_handle, line);
   FileClose(append_handle);

   ResetLastError();
   if(!FileMove(tmp_path, 0, final_path, FILE_REWRITE))
   {
      int err = GetLastError();
      PrintFormat("atomic_rename_failed | src=%s dst=%s code=%d", tmp_path, final_path, err);
      FileDelete(tmp_path);
      return false;
   }

   return true;
}
