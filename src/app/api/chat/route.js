export async function POST(request) {
  try {
    return Response.json(response)
  } catch (error) {
    console.error("API Route Error:", error)
    return Response.json(
      {
        introMessage: "Estoy teniendo algunos problemas técnicos. Por favor, inténtalo de nuevo en unos momentos.",
      },
      { status: 500 },
    )
  }
}