module Main

import Data.String

%hide Data.List.splitOn

-- Copy of Calc inline for compilation (RefC needs a main to generate .c)
abs' : Integer -> Integer
abs' n = if n < 0 then negate n else n

gcd' : Integer -> Integer -> Integer
gcd' a 0 = a
gcd' a b = gcd' b (a `mod` b)

breakOn : (Char -> Bool) -> List Char -> (List Char, List Char)
breakOn p [] = ([], [])
breakOn p (x :: xs) = 
  if p x then ([], x :: xs)
  else case breakOn p xs of
    (before, after) => (x :: before, after)

splitParts : Char -> List Char -> List (List Char)
splitParts sep cs = case breakOn (== sep) cs of
  (before, []) => [before]
  (before, _ :: after) => before :: splitParts sep after

parseRat : String -> Maybe (Integer, Integer)
parseRat s =
  let chars = unpack s in
  case splitParts '/' chars of
    [numChars] => 
      case parseInteger (pack numChars) of
        Just n => Just (n, 1)
        Nothing => Nothing
    [numChars, denChars] =>
      case (parseInteger (pack numChars), parseInteger (pack denChars)) of
        (Just n, Just d) => if d == 0 then Nothing else Just (n, d)
        _ => Nothing
    _ => Nothing

formatRat : (Integer, Integer) -> String
formatRat (num, 1) = show num
formatRat (num, den) = show num ++ "/" ++ show den

normalize : (Integer, Integer) -> (Integer, Integer)
normalize (num, den) =
  let g = gcd' (abs' num) (abs' den) in
  let num' = div num g in
  let den' = div den g in
  if den' < 0 then (negate num', negate den')
  else (num', den')

intPow : Integer -> Integer -> Integer
intPow base 0 = 1
intPow base exp = base * intPow base (exp - 1)

%export "C:add_rat"
add : String -> String
add input = 
  case splitParts ',' (unpack input) of
    [left, right] =>
      case (parseRat (pack left), parseRat (pack right)) of
        (Just (a, b), Just (c, d)) =>
          let num = a * d + c * b in
          let den = b * d in
          formatRat (normalize (num, den))
        _ => "PARSE_ERROR"
    _ => "PARSE_ERROR"

%export "C:sub_rat"
sub : String -> String
sub input = 
  case splitParts ',' (unpack input) of
    [left, right] =>
      case (parseRat (pack left), parseRat (pack right)) of
        (Just (a, b), Just (c, d)) =>
          let num = a * d - c * b in
          let den = b * d in
          formatRat (normalize (num, den))
        _ => "PARSE_ERROR"
    _ => "PARSE_ERROR"

%export "C:mul_rat"
mul : String -> String
mul input = 
  case splitParts ',' (unpack input) of
    [left, right] =>
      case (parseRat (pack left), parseRat (pack right)) of
        (Just (a, b), Just (c, d)) =>
          let num = a * c in
          let den = b * d in
          formatRat (normalize (num, den))
        _ => "PARSE_ERROR"
    _ => "PARSE_ERROR"

%export "C:div_rat"
div : String -> String
div input = 
  case splitParts ',' (unpack input) of
    [left, right] =>
      case (parseRat (pack left), parseRat (pack right)) of
        (Just (a, b), Just (c, d)) =>
          if c == 0 then "DIV_BY_ZERO"
          else
            let num = a * d in
            let den = b * c in
            formatRat (normalize (num, den))
        _ => "PARSE_ERROR"
    _ => "PARSE_ERROR"

%export "C:intpow_rat"
intpow : String -> String
intpow input = 
  case splitParts ',' (unpack input) of
    [ratStr, expStr] =>
      case (parseRat (pack ratStr), parseInteger (pack expStr)) of
        (Just (a, b), Just n) =>
          if n == 0 then "1"
          else if n > 0 then
            let num = intPow a n in
            let den = intPow b n in
            formatRat (normalize (num, den))
          else
            let m = negate n in
            let num = intPow b m in
            let den = intPow a m in
            formatRat (normalize (num, den))
        _ => "PARSE_ERROR"
    _ => "PARSE_ERROR"

main : IO ()
main = do
  putStrLn $ add "1/2,1/3"
  putStrLn $ sub "1/2,1/3"
  putStrLn $ mul "1/2,2/3"
  putStrLn $ div "1/2,2/3"
  putStrLn $ intpow "2/3,3"
  putStrLn $ intpow "2/3,-2"
  putStrLn $ div "1/2,0/1"
