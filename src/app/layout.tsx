import type { Metadata } from "next";
import { Poppins } from "next/font/google"; // Import Poppins
import "./globals.css";

// Configure the font
const poppins = Poppins({ 
  subsets: ["latin"],
  weight: ['400', '600', '700'] 
});

export const metadata: Metadata = {
  title: "PeepInMe - Costa Rica Store Finder",
  description: "A chatbot to help you find stores across Costa Rica.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      {/* Apply the font class to the body */}
      <body className={poppins.className}>{children}</body>
    </html>
  );
}