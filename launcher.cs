using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;

namespace TurtleScreenerLauncher
{
    class Program
    {
        static bool IsPortAvailable(int port)
        {
            try
            {
                TcpListener listener = new TcpListener(IPAddress.Loopback, port);
                listener.Start();
                listener.Stop();
                return true;
            }
            catch
            {
                return false;
            }
        }

        static int Main(string[] args)
        {
            Console.Title = "Turtle Trend Following US Stock Ranking Screener";
            string baseDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\', '/');
            string venvPython = Path.Combine(baseDir, ".venv", "Scripts", "python.exe");
            string pythonExe = File.Exists(venvPython) ? venvPython : "python";
            string screenerPy = Path.Combine(baseDir, "screener.py");
            string appPy = Path.Combine(baseDir, "app.py");
            string dashboardHtml = Path.Combine(baseDir, "output", "dashboard.html");

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("====================================================================================");
            Console.WriteLine("  TURTLE TREND FOLLOWING US STOCK RANKING SCREENER & IN-SYSTEM SPREADSHEET TRACKER");
            Console.WriteLine("====================================================================================");
            Console.ResetColor();

            if (!File.Exists(screenerPy))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("[ERROR] Could not find screener.py in: " + baseDir);
                Console.ResetColor();
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey();
                return 1;
            }

            bool noUi = false;
            string forwardedArgs = "";
            foreach (string a in args)
            {
                if (a.Equals("--no-ui", StringComparison.OrdinalIgnoreCase) ||
                    a.Equals("--no-browser", StringComparison.OrdinalIgnoreCase))
                {
                    noUi = true;
                }
                else
                {
                    forwardedArgs += " \"" + a.Replace("\"", "\\\"") + "\"";
                }
            }

            Console.WriteLine("[1/2] Running Daily EOD Sync, 5-Pillar Grading & Chart/Spreadsheet Generation...");
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = "\"" + screenerPy + "\"" + forwardedArgs,
                WorkingDirectory = baseDir,
                UseShellExecute = false
            };

            int exitCode = 0;
            using (Process proc = Process.Start(psi))
            {
                proc.WaitForExit();
                exitCode = proc.ExitCode;
            }

            if (exitCode != 0)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("\n[WARN] Online sync encountered an issue (exit code " + exitCode + "). Falling back to offline SQLite cache (--skip-sync)...");
                Console.ResetColor();

                ProcessStartInfo fallbackPsi = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = "\"" + screenerPy + "\" --skip-sync" + forwardedArgs,
                    WorkingDirectory = baseDir,
                    UseShellExecute = false
                };
                using (Process fallbackProc = Process.Start(fallbackPsi))
                {
                    fallbackProc.WaitForExit();
                    exitCode = fallbackProc.ExitCode;
                }
            }

            if (noUi)
            {
                return exitCode;
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\n[2/2] Opening Turtle Screener Web UI & Spreadsheets...");
            Console.ResetColor();

            if (File.Exists(appPy) && IsPortAvailable(8050))
            {
                Console.WriteLine("Starting local UI server at http://127.0.0.1:8050/dashboard.html (Press Ctrl+C to stop)...");
                ProcessStartInfo appPsi = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = "\"" + appPy + "\"",
                    WorkingDirectory = baseDir,
                    UseShellExecute = false
                };
                using (Process appProc = Process.Start(appPsi))
                {
                    appProc.WaitForExit();
                }
            }
            else if (File.Exists(dashboardHtml))
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = dashboardHtml,
                    UseShellExecute = true
                });
            }

            return exitCode;
        }
    }
}
