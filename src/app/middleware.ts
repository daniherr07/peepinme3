import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // Allow API routes to pass through without interference
  if (request.nextUrl.pathname.startsWith('/api')) {
    return NextResponse.next();
  }
  
}